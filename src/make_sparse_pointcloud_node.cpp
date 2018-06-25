// Subscribe to the compressed depth image and in the callback store a depth map
// Subscribe to the compressed rgb image and in the callback generate a 
//    sparse pointcloud using the rgb image + the latest depth map, then publish it
//
// Assumes rgb is BGR8 while depth is 16UC1

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
#include <rodan_vr_api/SparseXYZRGB.h>
#include <rodan_vr_api/CompressedSparsePointCloud.h>
#include <dynamic_reconfigure/server.h>
#include <zed_wrapper/MakePointcloudConfig.h>

// parameters for the special LZ compression
#define VERY_FAST 0
#define HLOG 22
#include "lzf.h"

boost::shared_ptr<dynamic_reconfigure::Server<zed_wrapper::MakePointcloudConfig>> server;

static int rate = 1;  // rate in Hz to publish pointcloud

namespace enc = sensor_msgs::image_encodings;
using namespace std;

typedef sensor_msgs::PointCloud2 PointCloud;
static ros::Publisher pub;
static int TotalPoints = 0;
static int Width = 0;
static int Height = 0;
static int seq = 0;
static rodan_vr_api::CompressedDepth Latest_depth_msg;
static bool HaveDepth = false;

rodan_vr_api::CompressedDepth CompressedDepth;
ros::Time LastDepthPublishTime = ros::Time(0);
static uint16_t *skrunchedDepth = nullptr;

int sparsepointdistsq = 625;
int sparsecolordistsq = 625;
int agelimit = 5;  // in frames

void callback(zed_wrapper::MakePointcloudConfig &config, uint32_t level) {
    ROS_INFO("Reconfigure: pointcloudrate %d", config.pointcloudrate);
    ROS_INFO("Reconfigure: sparsepointdist %d", config.sparsepointdist);
    ROS_INFO("Reconfigure: sparsecolordist %d", config.sparsecolordist);
    ROS_INFO("Reconfigure: agelimit %d", config.agelimit);
}

static int16_t zedToInt16(float v)
{
    // take the ZED value (float meters) and return int16 mm
    // limit values to +- 10000
    // if -inf, +inf, or NaN, or outside range, return -11111
    if (isnanf(v) || isinff(v)) {
       return -11111;
    }
    v *= 1000.0f;
    v += 0.5f;
    if (v > 10000.0f)  v=-11111.0f;
    if (v < -10000.0f) v=-11111.0f;
        return static_cast<int16_t>(v);
    }

static int dist3ds(int a1, int b1, int c1, int a2, int b2, int c2)
{
    return (a1 - a2) * (a1 - a2) + (b1 - b2) * (b1 - b2) + (c1 - c2) * (c1 - c2);
}

// take the rgb and depth images and construct the pointcloud
// then compress is using the sparse method but with parameters for LAN use
// for minimum "latency"
void convert(const rodan_vr_api::CompressedDepth& depth_msg,
             const cv::Mat rgb_image)
{
    static std::vector<rodan_vr_api::SparseXYZRGB> Baseline;  // baseline PC - not sent
    static std::vector<int32_t> BaselineLastUpdate; // frameNumber last updated
    static std::vector<rodan_vr_api::SparseXYZRGB> Updates;   // deltas from baseline
    static rodan_vr_api::SparseXYZRGB notvalid;
    static rodan_vr_api::CompressedSparsePointCloud CompressedUpdates;
    static unsigned int MaxCompressedSize;
    static int32_t TotalPoints = 0;
    static int32_t frameNumber = 0;

    frameNumber++;
    // total points is constant
    // check each point and only update ones that are enough different
    if (TotalPoints == 0) {
        TotalPoints = depth_msg.width * depth_msg.height;

        // reserve space for max for both vectors
        Baseline.reserve(TotalPoints);
        BaselineLastUpdate.reserve(TotalPoints);
        Updates.reserve(TotalPoints);
        MaxCompressedSize = LZF_MAX_COMPRESSED_SIZE(TotalPoints * sizeof(rodan_vr_api::SparseXYZRGB));
        CompressedUpdates.totalpoints = TotalPoints;
        CompressedUpdates.data.reserve(MaxCompressedSize);

        // init the notvalid entry
        notvalid.x = notvalid.y = notvalid.z = -11111;
        notvalid.r = notvalid.g = notvalid.b = 0;
        notvalid.index1 = notvalid.index2 = notvalid.index3 = 0;

        // initialize the entire baseline vector to notvalid
        for (int i = 0; i < TotalPoints; i++) {
            Baseline[i] = notvalid;
            BaselineLastUpdate[i] = 0;
        }
    } else {
        if (TotalPoints != (depth_msg.width * depth_msg.height)) {
            ROS_ERROR("You must restart make_sparse_pointcloud_node since the resolution was changed.");
        }
    }

    // go through and invalidate any entries older than age limit
    int lastValidFrame = frameNumber - agelimit;
    for (int i = 0; i < TotalPoints; i++) {
        if (BaselineLastUpdate[i] < lastValidFrame) {
            Baseline[i] = notvalid;
            BaselineLastUpdate[i] = 0;
        }
    }

    Updates.clear();  // No deltas yet

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

    if (!skrunchedDepth) {
        skrunchedDepth = (uint16_t *)malloc(depth_msg.width*depth_msg.height*sizeof(uint16_t));
    }

    // have a compressed depth_msg, first decompress to get the depth data
    unsigned int ucs = lzf_decompress(&depth_msg.data[0], 
                           depth_msg.data.size(),
                           skrunchedDepth, 
                           depth_msg.width*depth_msg.height*sizeof(uint16_t));

    int i = 0;
    for (int v = 0; v < depth_msg.height; ++v)
    {
        for (int u = 0; u < depth_msg.width; ++u, ++i, rgb += color_step)
        {
            uint16_t depth = skrunchedDepth[i];

            if (depth > 0) {   // don't generate pointcloud point for invalid
                // Fill in XYZ
                // map from u,v,depth to x,y,z using camera info
                float xx = (u - center_x) * depth * constant_x;
                float yy = (v - center_y) * depth * constant_y;
                float zz = depth * .001;  // convert to meters

                // need to reorder coords
                float nx = zz;
                float ny = -xx;
                float nz = -yy;

                // now apply the transform from the camera to rodan_vr_frame
                int16_t x = zedToInt16(nx * depth_msg.basis00 + 
                                ny * depth_msg.basis01 + 
                                nz * depth_msg.basis02 + depth_msg.originX);
                int16_t y = zedToInt16(nx * depth_msg.basis10 + 
                                ny * depth_msg.basis11 + 
                                nz * depth_msg.basis12 + depth_msg.originY);
                int16_t z = zedToInt16(nx * depth_msg.basis20 + 
                                ny * depth_msg.basis21 + 
                                nz * depth_msg.basis22 + depth_msg.originZ);

                uint8_t r = rgb[red_offset];
                uint8_t g = rgb[green_offset];
                uint8_t b = rgb[blue_offset];

                // see if different from Baseline, if so update it
                if ((dist3ds(x, y, z, Baseline[i].x, Baseline[i].y, Baseline[i].z) > sparsepointdistsq) ||
                    (dist3ds(r, g, b, Baseline[i].r, Baseline[i].g, Baseline[i].b) > sparsecolordistsq)) {

                    // update baseline with the new values
                    Baseline[i].x = x;
                    Baseline[i].y = y;
                    Baseline[i].z = z;
                    Baseline[i].r = r;
                    Baseline[i].g = g;
                    Baseline[i].b = b;
                    uint8_t* pp = (uint8_t*)&i;
                    Baseline[i].index1 = pp[0];
                    Baseline[i].index2 = pp[1];
                    Baseline[i].index3 = pp[2];
                    // and add the point to the sparse array
                    Updates.push_back(Baseline[i]);
                    BaselineLastUpdate[i] = frameNumber;
                }
            }
        }
    }
           
    // compress it
    CompressedUpdates.data.resize(MaxCompressedSize);  // first make sure we could store max size
    unsigned int cs = lzf_compress (&Updates[0], Updates.size() * sizeof(Updates[0]),
                                    &CompressedUpdates.data[0], MaxCompressedSize);
    CompressedUpdates.data.resize(cs);  // set it to proper compressed size
    pub.publish(CompressedUpdates);
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
    cv::Mat rgb_image = cv::imdecode(cv::Mat(rgb_msg->data),1);//convert compressed image data to cv::Mat
    convert(Latest_depth_msg, rgb_image);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "make_sparse_pointcloud_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    
    ros::Subscriber depthSub = nh.subscribe<rodan_vr_api::CompressedDepth>("/zed/depth/compressed_depth_svrt", 1, depthCb);

    pub = nh.advertise<rodan_vr_api::CompressedSparsePointCloud>("/zed/point_cloud/compressed", 1);
    // we need to sleep until we have the initial depth callback
    while (!HaveDepth) {
        sleep(1);
        ros::spinOnce();
    }

    ros::Subscriber rgbSub = 
        nh.subscribe<sensor_msgs::CompressedImageConstPtr>("/zed/rgb/image_rect_color/compressed", 1, rgbCb);

    //Reconfigure for various parameters
    server = boost::make_shared<dynamic_reconfigure::Server<zed_wrapper::MakePointcloudConfig>>();
    dynamic_reconfigure::Server<zed_wrapper::MakePointcloudConfig>::CallbackType f;
    f = boost::bind(&callback, _1, _2);
    server->setCallback(f);

    while(true) { 
        ros::Rate(rate).sleep(); 
        ros::spinOnce();
    }

    return 0;
}
