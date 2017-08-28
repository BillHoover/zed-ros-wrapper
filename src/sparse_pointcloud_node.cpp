#include <cstdio>
#include <ros/ros.h>
#include <rodan_vr_api/SparseXYZRGB.h>
#include <rodan_vr_api/CompressedSparsePointCloud.h>
#include "pcl_ros/point_cloud.h"
#include "lzf.h"

static ros::Publisher pub;
static pcl::PointCloud<pcl::PointXYZRGB> cloud;
static std::vector<rodan_vr_api::SparseXYZRGB> Updates;   // deltas from baseline
static int TotalPoints = 0;

static float zedFromInt16(int16_t v)
{
    // return the ZED value (float meters) from int16 mm
    // if int16 value was -11111, return NAN
    if (v == -11111) return NAN;
    return v / 1000.0f;
}

void cb(const rodan_vr_api::CompressedSparsePointCloud compressed)
{
    // if this is the first time called, init some things
    if (TotalPoints == 0) {
        TotalPoints = compressed.totalpoints;
        Updates.reserve(TotalPoints);  // space for entire map
        cloud.resize(TotalPoints);  // always this many points
        // and update to all nan
        for (int i = 0; i < TotalPoints; i++) {
            cloud[i].x = cloud[i].y = cloud[i].z = NAN;
            cloud[i].r = cloud[i].g = cloud[i].b = 0;
        }
    }

    // have a compressed point cloud, first decompress to get the sparse data
    Updates.resize(TotalPoints);  // first make sure we could store max size
    unsigned int ucs = lzf_decompress(&compressed.data[0], compressed.data.size(),
                                      &Updates[0], Updates.size() * sizeof(Updates[0]));
    
    for (int n = 0; n < ucs/sizeof(Updates[0]); n++) {
        // need to construct the point pointer from the three bytes 
        uint32_t i = 0;
        uint8_t* pp = (uint8_t*)&i;
        pp[0] = Updates[n].index1;
        pp[1] = Updates[n].index2;
        pp[2] = Updates[n].index3;
        cloud[i].x = zedFromInt16(Updates[n].x);
        cloud[i].y = zedFromInt16(Updates[n].y);
        cloud[i].z = zedFromInt16(Updates[n].z);
        cloud[i].r = Updates[n].r;
        cloud[i].g = Updates[n].g;
        cloud[i].b = Updates[n].b;
    }
    pub.publish(cloud);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "sparse_pointcloud_node");
    ros::NodeHandle nh;
    
    ros::Subscriber sub = 
        nh.subscribe<rodan_vr_api::CompressedSparsePointCloud>("compressedPointcloud", 10, cb);
    pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("pointcloud", 1);

    ros::spin();

    return 0;
}
