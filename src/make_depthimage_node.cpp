// Subscribe to the compressed depthimage and publish a normal depthimage
// For the depthimage_to_laserscan node we need it synchronized with /zed/depth/camera_info
#include <cstdio>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <rodan_vr_api/CompressedDepth.h>
#include "lzf.h"

ros::Publisher pub_depthimage_;
static ros::Publisher pub;
static int TotalPoints = 0;
static int Width = 0;
static int Height = 0;
static bool HaveDepth = false;
static sensor_msgs::Image depthimage;

static sensor_msgs::CameraInfo LatestCameraInfo;
void cameraInfoCb(const sensor_msgs::CameraInfo camera_info_msg)
{
    LatestCameraInfo = camera_info_msg;
}

void depthCb(const rodan_vr_api::CompressedDepth depth_msg)
{
    // if this is the first time called, init some things
    if (!HaveDepth) {
        Width = depth_msg.width;
        Height = depth_msg.height;
        TotalPoints = Width * Height;
        HaveDepth = true;
        // set up the constant fields in the image record
        depthimage.height = Height;
        depthimage.width = Width;
        depthimage.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
        depthimage.is_bigendian = 0;
        depthimage.step = Width * sizeof(uint16_t);
        size_t size = depthimage.step * Height;
        depthimage.data.resize(size);
        depthimage.header.frame_id = "";
    }

  // have a compressed depth_msg, first decompress to get the depth data
  unsigned int ucs = lzf_decompress(&depth_msg.data[0], 
                         depth_msg.data.size(),
                         &depthimage.data[0], 
                         depth_msg.height * depth_msg.width * sizeof(uint16_t));


    depthimage.header.stamp = LatestCameraInfo.header.stamp; 
    pub.publish(depthimage);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "make_depthimage_node");
    ros::NodeHandle nh;
    
    ros::Subscriber cameraInfoSub = nh.subscribe<sensor_msgs::CameraInfo>("/zed/depth/camera_info", 1, cameraInfoCb);
    ros::Subscriber depthSub = nh.subscribe<rodan_vr_api::CompressedDepth>("/zed/depth/compressed_depth_svrt", 1, depthCb);

    pub = nh.advertise<sensor_msgs::Image>("/zed/depth/depth", 1);

    ros::spin();

    return 0;
}
