// Subscribe to the camera info to get the parameters for the Zed
//    Once we have them, we can unsubcribe to it as it won't change
// Subscribe to the compressed depth image and in the callback store a depth map
// Subscribe to the compressed rgb image and in the callback generate a 
//    pointcloud using the rgb image + the latest depth map, then publish is
//
// Assumes rgb iss BGR8 while depth is 16UC1

#include <cstdio>
#include <ros/ros.h>
#include "pcl_ros/point_cloud.h"
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_geometry/pinhole_camera_model.h>
#include <depth_image_proc/depth_traits.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>

namespace enc = sensor_msgs::image_encodings;

typedef sensor_msgs::PointCloud2 PointCloud;
ros::Publisher pub_point_cloud_;
image_geometry::PinholeCameraModel model_;
static ros::Publisher pub;
static ros::Subscriber infoSub;
static int TotalPoints = 0;
static int Width = 0;
static int Height = 0;
static int seq = 0;
static sensor_msgs::ImageConstPtr Latest_depth_msg;

void convert(const sensor_msgs::ImageConstPtr& depth_msg,
             const cv::Mat rgb_image,
             const PointCloud::Ptr& cloud_msg)
{
  // Use correct principal point from calibration
  float center_x = model_.cx();
  float center_y = model_.cy();

  // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
  double unit_scaling = depth_image_proc::DepthTraits<uint16_t>::toMeters( uint16_t(1) );
  float constant_x = unit_scaling / model_.fx();
  float constant_y = unit_scaling / model_.fy();
  float bad_point = std::numeric_limits<float>::quiet_NaN ();
  
  const int red_offset   = 2;
  const int green_offset = 1;
  const int blue_offset  = 0;
  const int color_step   = 3;

  const uint16_t* depth_row = reinterpret_cast<const uint16_t*>(&depth_msg->data[0]);
  int row_step = depth_msg->step / sizeof(uint16_t);
  const uint8_t* rgb = &rgb_image.at<uint8_t>(0, 0);

  sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*cloud_msg, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*cloud_msg, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*cloud_msg, "b");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_a(*cloud_msg, "a");

  for (int v = 0; v < int(cloud_msg->height); ++v, depth_row += row_step)
  {
    for (int u = 0; u < int(cloud_msg->width); ++u, rgb += color_step, ++iter_x, ++iter_y, ++iter_z, ++iter_a, ++iter_r, ++iter_g, ++iter_b)
    {
      uint16_t depth = depth_row[u];

      // Check for invalid measurements
      if (!depth_image_proc::DepthTraits<uint16_t>::valid(depth))
      {
        *iter_x = *iter_y = *iter_z = bad_point;
      }
      else
      {
        // Fill in XYZ
        *iter_x = (u - center_x) * depth * constant_x;
        *iter_y = (v - center_y) * depth * constant_y;
        *iter_z = depth_image_proc::DepthTraits<uint16_t>::toMeters(depth);
      }

      // Fill in color
      *iter_a = 255;
      *iter_r = rgb[red_offset];
      *iter_g = rgb[green_offset];
      *iter_b = rgb[blue_offset];
    }
  }
}

void infoCb(const sensor_msgs::CameraInfoConstPtr info_msg)
{
  // unsubscribe since the parameters won't change
  infoSub.shutdown();  
  // Update camera model
  model_.fromCameraInfo(info_msg);
}

void depthCb(const sensor_msgs::ImageConstPtr depth_msg)
{
    // if this is the first time called, init some things
    if (TotalPoints == 0) {
        if (depth_msg->encoding != enc::TYPE_16UC1) {
            ROS_ERROR("Incorrect depth encoding");
            abort();
        }
        Width = depth_msg->width;
        Height = depth_msg->height;
        TotalPoints = Width * Height;
    }

    Latest_depth_msg = depth_msg;
}

void rgbCb(const sensor_msgs::CompressedImageConstPtr rgb_msg)
{
    // Allocate new point cloud message
    PointCloud::Ptr Cloud_msg(new PointCloud);
    Cloud_msg->height = Latest_depth_msg->height;
    Cloud_msg->width  = Latest_depth_msg->width;
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
    
    infoSub = 
        nh.subscribe<sensor_msgs::CameraInfoConstPtr>("/zed/depth/camera_info", 1, infoCb);
    image_transport::Subscriber depthSub = it.subscribe("/zed/depth/depth_registered", 1, depthCb);

    pub = nh.advertise<PointCloud>("pointcloud", 1);
    sleep(1);  // not the right way, want to make sure get info and depth

    ros::Subscriber rgbSub = 
        nh.subscribe<sensor_msgs::CompressedImageConstPtr>("/zed/rgb/image_rect_color/compressed", 1, rgbCb);

    ros::spin();

    return 0;
}
