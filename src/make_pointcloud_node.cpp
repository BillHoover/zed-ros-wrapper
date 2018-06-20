// Subscribe to the compressed depth image and in the callback store a depth map
// Subscribe to the compressed rgb image and in the callback generate a 
//    pointcloud using the rgb image + the latest depth map, then publish it
//
// Assumes rgb is BGR8 while depth is 16UC1
// Things are done in simpler ways as needs to be duplicated in C# Unity code

#include <cstdio>
#include <ros/ros.h>
#include "pcl_ros/point_cloud.h"
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <rodan_vr_api/CompressedDepth.h>
#include "lzf.h"

namespace enc = sensor_msgs::image_encodings;

typedef sensor_msgs::PointCloud2 PointCloud;
ros::Publisher pub_point_cloud_;
static ros::Publisher pub;
static int TotalPoints = 0;
static int Width = 0;
static int Height = 0;
static int seq = 0;
static rodan_vr_api::CompressedDepth Latest_depth_msg;
static bool HaveDepth = false;

void convert(const rodan_vr_api::CompressedDepth& depth_msg,
             const cv::Mat rgb_image,
             const PointCloud::Ptr& cloud_msg)
{
  // Use correct principal point from calibration
  float center_x = depth_msg.cx;
  float center_y = depth_msg.cy;

  // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
  double unit_scaling = .001;  //convert mm to m
  float constant_x = unit_scaling / depth_msg.fx;
  float constant_y = unit_scaling / depth_msg.fy;
  
  const int red_offset   = 2;
  const int green_offset = 1;
  const int blue_offset  = 0;
  const int color_step   = 3;

  const uint8_t* rgb = &rgb_image.at<uint8_t>(0, 0);

  sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*cloud_msg, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*cloud_msg, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*cloud_msg, "b");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_a(*cloud_msg, "a");

  // have a compressed depth_msg, first decompress to get the depth data
  static uint16_t *skrunchedDepth = nullptr;
  if (!skrunchedDepth) {
      skrunchedDepth = (uint16_t *)malloc(cloud_msg->height*cloud_msg->width*sizeof(uint16_t));
  }
  unsigned int ucs = lzf_decompress(&depth_msg.data[0], 
                         depth_msg.data.size(),
                         skrunchedDepth, 
                         cloud_msg->height * cloud_msg->width * sizeof(uint16_t));

  int i = 0;
  for (int v = 0; v < int(cloud_msg->height); ++v)
  {
    for (int u = 0; u < int(cloud_msg->width); ++u, ++i, rgb += color_step, ++iter_x, ++iter_y, ++iter_z, ++iter_a, ++iter_r, ++iter_g, ++iter_b)
    {
      uint16_t depth = skrunchedDepth[i];

      if (depth > 0) {   // don't generate pointcloud point for invalid
        // Fill in XYZ
        // map from u,v,depth to x,y,z using camera info
        float x = (u - center_x) * depth * constant_x;
        float y = (v - center_y) * depth * constant_y;
        float z = depth * .001;  // convert to meters

        // need to reorder coords
        float nx = z;
        float ny = -x;
        float nz = -y;

        // now apply the transform from the camera to rodan_vr_frame
        *iter_x = nx * depth_msg.basis00 + 
                  ny * depth_msg.basis01 + 
                  nz * depth_msg.basis02 + depth_msg.originX;
        *iter_y = nx * depth_msg.basis10 + 
                  ny * depth_msg.basis11 + 
                  nz * depth_msg.basis12 + depth_msg.originY;
        *iter_z = nx * depth_msg.basis20 + 
                  ny * depth_msg.basis21 + 
                  nz * depth_msg.basis22 + depth_msg.originZ;

        // Fill in color
        *iter_a = 255;
        *iter_r = rgb[red_offset];
        *iter_g = rgb[green_offset];
        *iter_b = rgb[blue_offset];
      }
    }
  }
}

void depthCb(const rodan_vr_api::CompressedDepth depth_msg)
{
    // if this is the first time called, init some things
    if (!HaveDepth) {
        Width = depth_msg.width;
        Height = depth_msg.height;
        TotalPoints = Width * Height;
        HaveDepth = true;
    }

    Latest_depth_msg = depth_msg;
}

void rgbCb(const sensor_msgs::CompressedImageConstPtr rgb_msg)
{
    // Allocate new point cloud message
    PointCloud::Ptr Cloud_msg(new PointCloud);
    Cloud_msg->height = Latest_depth_msg.height;
    Cloud_msg->width  = Latest_depth_msg.width;
    Cloud_msg->is_dense = false;
    Cloud_msg->is_bigendian = false;

    cv::Mat rgb_image = cv::imdecode(cv::Mat(rgb_msg->data),1);//convert compressed image data to cv::Mat
    sensor_msgs::PointCloud2Modifier pcd_modifier(*Cloud_msg);
    pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    convert(Latest_depth_msg, rgb_image, Cloud_msg);
    Cloud_msg->header.seq = seq++;
    Cloud_msg->header.stamp = rgb_msg->header.stamp;  // use rgb timestamp
    Cloud_msg->header.frame_id = "rodan_vr_frame";
    pub.publish(Cloud_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "make_pointcloud_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    
    ros::Subscriber depthSub = nh.subscribe<rodan_vr_api::CompressedDepth>("/zed/depth/compressed_depth_svrt", 1, depthCb);

    pub = nh.advertise<PointCloud>("/zed/pointcloud_svrt", 1);
    // we need to sleep until we have the initial depth callback
    while (!HaveDepth) {
        sleep(1);
        ros::spinOnce();
    }

    ros::Subscriber rgbSub = 
        nh.subscribe<sensor_msgs::CompressedImageConstPtr>("/zed/rgb/image_rect_color/compressed", 1, rgbCb);

    ros::spin();

    return 0;
}
