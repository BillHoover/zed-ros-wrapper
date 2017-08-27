#include <ros/ros.h>
#include <rodan_vr_api/SparseXYZRGB.h>
#include <rodan_vr_api/CompressedSparsePointCloud.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "sparse_pointcloud_node");

    ros::spin();

    return 0;
}

/*

#include <csignal>
#include <cstdio>
#include <math.h>
#include <limits>
#include <thread>
#include <chrono>
#include <memory>
#include <sys/stat.h>

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/distortion_models.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <dynamic_reconfigure/server.h>
#include <zed_wrapper/ZedConfig.h>
#include <nav_msgs/Odometry.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <boost/make_shared.hpp>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <sl/Camera.hpp>

        ros::NodeHandle nh;
        ros::NodeHandle nh_ns;
        ros::Publisher pub_cloud;
        // Point cloud variables
        string point_cloud_frame_id = "";
        ros::Time point_cloud_time;

                    // Publish the point cloud if someone has subscribed to
                    if (cloud_SubNumber > 0) {
                        // Run the point cloud convertion asynchronously to avoid slowing down all the program
                        // Retrieve raw pointCloud data
                        zed->retrieveMeasure(cloud, sl::MEASURE_XYZBGRA);
                        point_cloud_frame_id = cloud_frame_id;
                        point_cloud_time = t;
                        publishPointCloud(width, height, pub_cloud);
                    }

        boost::shared_ptr<dynamic_reconfigure::Server<zed_wrapper::ZedConfig>> server;
            nh = getMTNodeHandle();
            nh_ns = getMTPrivateNodeHandle();

            //PointCloud publisher
            pub_cloud = nh.advertise<sensor_msgs::PointCloud2> (point_cloud_topic, 1);
*/
