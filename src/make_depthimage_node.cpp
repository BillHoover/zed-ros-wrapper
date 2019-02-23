// Subscribe to the compressed depthimage and publish a normal depthimage
#include <cstdio>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <rodan_vr_api/CompressedDepth.h>
#include "lzf.h"

ros::Publisher pub_depthimage_;
static ros::Publisher pub;
static int TotalPoints = 0;
static int Width = 0;
static int Height = 0;
static bool HaveDepth = false;
static sensor_msgs::Image depthimage_msg;

void depthCb(const rodan_vr_api::CompressedDepth depth_msg)
{
  static uint16_t *skrunchedDepth = nullptr;
    // if this is the first time called, init some things
    if (!HaveDepth) {
        Width = depth_msg.width;
        Height = depth_msg.height;
        TotalPoints = Width * Height;
        HaveDepth = true;
        skrunchedDepth = (uint16_t *)malloc(depth_msg.height*depth_msg.width*sizeof(uint16_t));
    }

  // have a compressed depth_msg, first decompress to get the depth data
  unsigned int ucs = lzf_decompress(&depth_msg.data[0], 
                         depth_msg.data.size(),
                         skrunchedDepth, 
                         depth_msg.height * depth_msg.width * sizeof(uint16_t));

  // need to fill in header and data
  //depthimage_msg.header = ;
    int i = 0;
      uint16_t depth = skrunchedDepth[i];

    pub.publish(depthimage_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "make_depthimage_node");
    ros::NodeHandle nh;
    
    ros::Subscriber depthSub = nh.subscribe<rodan_vr_api::CompressedDepth>("/zed/depth/compressed_depth_svrt", 1, depthCb);

    pub = nh.advertise<sensor_msgs::Image>("/zed/depth/depth", 1);

    ros::spin();

    return 0;
}
