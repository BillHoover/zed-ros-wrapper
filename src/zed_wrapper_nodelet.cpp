///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////


/****************************************************************************************************
 ** This sample is a wrapper for the ZED library in order to use the ZED Camera with ROS.          **
 ** A set of parameters can be specified in the launch file.                                       **
 ** Modified for SVRT use, several things with depth maps and pointclouds
 ** changed to be various hard-coded options
 ****************************************************************************************************/

#include <csignal>
#include <cstdio>
#include <math.h>
#include <limits>
#include <thread>
#include <chrono>
#include <memory>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/cuda.hpp>

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/distortion_models.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <dynamic_reconfigure/server.h>
#include <zed_wrapper/ZedConfig.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Vector3.h>

#include <boost/make_shared.hpp>
#include <boost/thread.hpp>

#include <sl_zed/Camera.hpp>

#include <rodan_vr_api/CompressedDepth.h>

// parameters for the special LZ compression
#define VERY_FAST 0
#define HLOG 22
#include "lzf.h"

static uint16_t *skrunchedDepth = nullptr;
static uint16_t *lastDepth = nullptr;
static uint8_t *lastDepthAge = nullptr;

using namespace std;
using namespace cv::cuda;

namespace zed_wrapper {

    class ZEDWrapperNodelet : public nodelet::Nodelet {
        ros::NodeHandle nh;
        ros::NodeHandle nh_ns;
        boost::shared_ptr<boost::thread> device_poll_thread;
        image_transport::Publisher pub_rgb;
        image_transport::Publisher pub_raw_rgb;
        image_transport::Publisher pub_left;
        image_transport::Publisher pub_raw_left;
        image_transport::Publisher pub_right;
        image_transport::Publisher pub_raw_right;
        ros::Publisher pub_depth;
        ros::Publisher pub_rgb_cam_info;
        ros::Publisher pub_left_cam_info;
        ros::Publisher pub_right_cam_info;
        ros::Publisher pub_depth_cam_info;

        // initialization Transform listener
        boost::shared_ptr<tf2_ros::Buffer> tfBuffer;
        boost::shared_ptr<tf2_ros::TransformListener> tf_listener;

        // Launch file parameters
        int resolution;
        int quality;
        int sensing_mode;
        int rate;
        int gpu_id;
        int zed_id;
        int depth_stabilization;

        rodan_vr_api::CompressedDepth CompressedDepth;
        ros::Time LastDepthPublishTime = ros::Time(0);

        // zed object
        sl::InitParameters param;
        sl::Camera zed;

        // flags
        int confidence = 100;

        bool computeDepth;  // whether to compute depth
        bool computingDepth = false;  // whether we currently are computing depth
        bool grabbing = false;

        // Transform from zed_initial_frame to rodan_vr_frame
        tf2::Transform camera_to_vr;  // will always have the latest valid one

        string depth_frame_id = "";
        string camera_frame_id = "";
        string rgb_frame_id = "";
        string right_frame_id = "";
        string left_frame_id = "";

        ~ZEDWrapperNodelet() {
            device_poll_thread.get()->join();
        }

        /* \brief Convert an sl:Mat to a cv::Mat
         * \param mat : the sl::Mat to convert
         */
        cv::Mat toCVMat(sl::Mat &mat) {
            if (mat.getMemoryType() == sl::MEM_GPU)
                mat.updateCPUfromGPU();

            int cvType;
            switch (mat.getDataType()) {
                case sl::MAT_TYPE_32F_C1:
                    cvType = CV_32FC1;
                    break;
                case sl::MAT_TYPE_32F_C2:
                    cvType = CV_32FC2;
                    break;
                case sl::MAT_TYPE_32F_C3:
                    cvType = CV_32FC3;
                    break;
                case sl::MAT_TYPE_32F_C4:
                    cvType = CV_32FC4;
                    break;
                case sl::MAT_TYPE_8U_C1:
                    cvType = CV_8UC1;
                    break;
                case sl::MAT_TYPE_8U_C2:
                    cvType = CV_8UC2;
                    break;
                case sl::MAT_TYPE_8U_C3:
                    cvType = CV_8UC3;
                    break;
                case sl::MAT_TYPE_8U_C4:
                    cvType = CV_8UC4;
                    break;
            }
            return cv::Mat((int) mat.getHeight(), (int) mat.getWidth(), cvType, mat.getPtr<sl::uchar1>(sl::MEM_CPU), mat.getStepBytes(sl::MEM_CPU));
        }

        /* \brief Image to ros message conversion
         * \param img : the image to publish
         * \param encodingType : the sensor_msgs::image_encodings encoding type
         * \param frameId : the id of the reference frame of the image
         * \param t : the ros::Time to stamp the image
         */
        sensor_msgs::ImagePtr imageToROSmsg(cv::Mat img, const std::string encodingType, std::string frameId, ros::Time t) {
            sensor_msgs::ImagePtr ptr = boost::make_shared<sensor_msgs::Image>();
            sensor_msgs::Image& imgMessage = *ptr;
            imgMessage.header.stamp = t;
            imgMessage.header.frame_id = frameId;
            imgMessage.height = img.rows;
            imgMessage.width = img.cols;
            imgMessage.encoding = encodingType;
            int num = 1; //for endianness detection
            imgMessage.is_bigendian = !(*(char *) &num == 1);
            imgMessage.step = img.cols * img.elemSize();
            size_t size = imgMessage.step * img.rows;
            imgMessage.data.resize(size);

            if (img.isContinuous())
                memcpy((char*) (&imgMessage.data[0]), img.data, size);
            else {
                uchar* opencvData = img.data;
                uchar* rosData = (uchar*) (&imgMessage.data[0]);
                for (unsigned int i = 0; i < img.rows; i++) {
                    memcpy(rosData, opencvData, imgMessage.step);
                    rosData += imgMessage.step;
                    opencvData += img.step;
                }
            }
            return ptr;
        }

        /* \brief Publish a cv::Mat image with a ros Publisher
         * \param img : the image to publish
         * \param pub_img : the publisher object to use
         * \param img_frame_id : the id of the reference frame of the image
         * \param t : the ros::Time to stamp the image
         */
        void publishImage(cv::Mat img, image_transport::Publisher &pub_img, string img_frame_id, ros::Time t) {
            pub_img.publish(imageToROSmsg(img, sensor_msgs::image_encodings::BGR8, img_frame_id, t));
        }

        // clamp the depth value between .3m and 5m and convert to uint16 mm
        static uint16_t dv(float v) {
            if (isnanf(v)) {
                v = 0.0;
            } else {
                if (isinff(v)) {
                    if (v < 0.0) v = .3;
                    else  v = 5.0;
                }
            }
            return (uint16_t)(v * 1000.0f + .5f);
        }

        /* \brief Publish a cv::Mat depth image with a ros Publisher
         * \param depth : the depth image to publish
         * \param pub_depth : the publisher object to use
         * \param depth_frame_id : the id of the reference frame of the depth image
         * \param t : the ros::Time to stamp the depth image
         */
        void publishDepth(cv::Mat depth, ros::Publisher &pub_depth, string depth_frame_id, ros::Time t, int width, int height) {

            size_t skrunchedDepthAlloc = width*height*sizeof(uint16_t);
            if (!skrunchedDepth) {
                skrunchedDepth = (uint16_t *)malloc(skrunchedDepthAlloc);
                lastDepth = (uint16_t *)malloc(skrunchedDepthAlloc);
                lastDepthAge = (uint8_t *)malloc(width*height*sizeof(uint8_t));
                for (int i = 0; i < width*height; i++) lastDepthAge[i] = 255;
            }
            int pixelsold = 0;
            int pixelsbad = 0;
            int pixelsnew = 0;
            int pixelsadj = 0;
            int i = 0;
            for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++, ++i) {
                skrunchedDepth[i] = dv(depth.at<float>(row, col));
                if (skrunchedDepth[i] == 0) {  // no new value
                    if (lastDepthAge[i] < 255) { // if a recent enough value use it
                        lastDepthAge[i] += 1;
                        skrunchedDepth[i] = lastDepth[i];
                        pixelsold++;
                    } else {
                       pixelsbad++;
                    }
                } else {
                    lastDepth[i] = skrunchedDepth[i];
                    lastDepthAge[i] = 0;
                    pixelsnew++;
                }
            }}

            // make a pass and try using adjacent for any remaining bad ones
            // do it from lastDepth (which at this point is current) to avoid cascading things
            i = 0;
            for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++, ++i) {
                if (skrunchedDepth[i] != 0) continue;  // have a good value so nothing to do
                // also skip the edges as we don't have enough adjacent pixels
                if ((col < 1) || (col > width-2) || (row < 1) || (row > height-2)) continue;

                int avgpixel = 0;
                int goodadj = 0;
                // note that we know this one has 0, so can blindly "include" below
                for (int k = i-1; k <= i+1; k++) {
                    if (lastDepth[k - width] > 0) {
                        avgpixel += lastDepth[k - width]; // above   
                        ++goodadj;
                    }
                    if (lastDepth[k] > 0) {
                        avgpixel += lastDepth[k]; // in line   
                        ++goodadj;
                    }
                    if (lastDepth[k + width] > 0) {
                        avgpixel += lastDepth[k + width]; // below   
                        ++goodadj;
                    }
                }
                // if we have at least two valid adj pixels, use it
                if (goodadj >= 2) {
                    skrunchedDepth[i] = avgpixel / goodadj; 
                    --pixelsbad;
                    ++pixelsadj;
                }
            }}
            //NODELET_INFO_STREAM("Pixels: new: " << pixelsnew << ", old: " << pixelsold << ", adj: " << pixelsadj << ", bad: " << pixelsbad);
            

            // Now compress it
            const unsigned int MaxCompressedSize =
                LZF_MAX_COMPRESSED_SIZE(skrunchedDepthAlloc);
            CompressedDepth.data.resize(MaxCompressedSize);  // first make sure we could store max size
            unsigned int cs = lzf_compress (skrunchedDepth, skrunchedDepthAlloc,
                                            &CompressedDepth.data[0], MaxCompressedSize);
            CompressedDepth.data.resize(cs);  // set it to proper compressed size
            CompressedDepth.width = width;
            CompressedDepth.height = height;

            // add camera info so we don't need access in windows
            sl::CameraInformation zedParam = zed.getCameraInformation();
            CompressedDepth.fx = zedParam.calibration_parameters.left_cam.fx;
            CompressedDepth.fy = zedParam.calibration_parameters.left_cam.fy;
            CompressedDepth.cx = zedParam.calibration_parameters.left_cam.cx;
            CompressedDepth.cy = zedParam.calibration_parameters.left_cam.cy;

            // add current transform to rodan_vr_frame so easy in windows
            // Transform from zed_initial_frame to rodan_vr_frame
            // was initialized to identity in case we don't get a transform
            // if for some reason TF fails, will use the last valid one
            try {
                geometry_msgs::TransformStamped c2v = 
                    tfBuffer->lookupTransform("rodan_vr_frame", "zed_initial_frame", 
                                              ros::Time(0)); // use zero for latest tf
                tf2::fromMsg(c2v.transform, camera_to_vr);
            } catch (...) {}  // ugly, but really just want to leave the latest transform on error
            CompressedDepth.originX = camera_to_vr.getOrigin()[0];
            CompressedDepth.originY = camera_to_vr.getOrigin()[1];
            CompressedDepth.originZ = camera_to_vr.getOrigin()[2];
            CompressedDepth.basis00 = camera_to_vr.getBasis()[0][0];
            CompressedDepth.basis01 = camera_to_vr.getBasis()[0][1];
            CompressedDepth.basis02 = camera_to_vr.getBasis()[0][2];
            CompressedDepth.basis10 = camera_to_vr.getBasis()[1][0];
            CompressedDepth.basis11 = camera_to_vr.getBasis()[1][1];
            CompressedDepth.basis12 = camera_to_vr.getBasis()[1][2];
            CompressedDepth.basis20 = camera_to_vr.getBasis()[2][0];
            CompressedDepth.basis21 = camera_to_vr.getBasis()[2][1];
            CompressedDepth.basis22 = camera_to_vr.getBasis()[2][2];

            LastDepthPublishTime = ros::Time::now();
            pub_depth.publish(CompressedDepth);
        }


        /* \brief Publish the informations of a camera with a ros Publisher
         * \param cam_info_msg : the information message to publish
         * \param pub_cam_info : the publisher object to use
         * \param t : the ros::Time to stamp the message
         */
        void publishCamInfo(sensor_msgs::CameraInfoPtr cam_info_msg, ros::Publisher pub_cam_info, ros::Time t) {
            static int seq = 0;
            cam_info_msg->header.stamp = t;
            cam_info_msg->header.seq = seq;
            pub_cam_info.publish(cam_info_msg);
            seq++;
        }

        /* \brief Get the information of the ZED cameras and store them in an information message
         * \param zed : the sl::zed::Camera* pointer to an instance
         * \param left_cam_info_msg : the information message to fill with the left camera informations
         * \param right_cam_info_msg : the information message to fill with the right camera informations
         * \param left_frame_id : the id of the reference frame of the left camera
         * \param right_frame_id : the id of the reference frame of the right camera
         */
        void fillCamInfo(sl::Camera& zed, sensor_msgs::CameraInfoPtr left_cam_info_msg, sensor_msgs::CameraInfoPtr right_cam_info_msg,
                string left_frame_id, string right_frame_id) {

            int width = zed.getResolution().width;
            int height = zed.getResolution().height;

            sl::CameraInformation zedParam = zed.getCameraInformation();

            float baseline = zedParam.calibration_parameters.T[0];

            float fx = zedParam.calibration_parameters.left_cam.fx;
            float fy = zedParam.calibration_parameters.left_cam.fy;
            float cx = zedParam.calibration_parameters.left_cam.cx;
            float cy = zedParam.calibration_parameters.left_cam.cy;

            // There is no distortions since the images are rectified
            double k1 = 0;
            double k2 = 0;
            double k3 = 0;
            double p1 = 0;
            double p2 = 0;

            left_cam_info_msg->distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;
            right_cam_info_msg->distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;

            left_cam_info_msg->D.resize(5);
            right_cam_info_msg->D.resize(5);
            left_cam_info_msg->D[0] = right_cam_info_msg->D[0] = k1;
            left_cam_info_msg->D[1] = right_cam_info_msg->D[1] = k2;
            left_cam_info_msg->D[2] = right_cam_info_msg->D[2] = k3;
            left_cam_info_msg->D[3] = right_cam_info_msg->D[3] = p1;
            left_cam_info_msg->D[4] = right_cam_info_msg->D[4] = p2;

            left_cam_info_msg->K.fill(0.0);
            right_cam_info_msg->K.fill(0.0);
            left_cam_info_msg->K[0] = right_cam_info_msg->K[0] = fx;
            left_cam_info_msg->K[2] = right_cam_info_msg->K[2] = cx;
            left_cam_info_msg->K[4] = right_cam_info_msg->K[4] = fy;
            left_cam_info_msg->K[5] = right_cam_info_msg->K[5] = cy;
            left_cam_info_msg->K[8] = right_cam_info_msg->K[8] = 1.0;

            left_cam_info_msg->R.fill(0.0);
            right_cam_info_msg->R.fill(0.0);

            left_cam_info_msg->P.fill(0.0);
            right_cam_info_msg->P.fill(0.0);
            left_cam_info_msg->P[0] = right_cam_info_msg->P[0] = fx;
            left_cam_info_msg->P[2] = right_cam_info_msg->P[2] = cx;
            left_cam_info_msg->P[5] = right_cam_info_msg->P[5] = fy;
            left_cam_info_msg->P[6] = right_cam_info_msg->P[6] = cy;
            left_cam_info_msg->P[10] = right_cam_info_msg->P[10] = 1.0;
            right_cam_info_msg->P[3] = (-1 * fx * baseline);

            left_cam_info_msg->width = right_cam_info_msg->width = width;
            left_cam_info_msg->height = right_cam_info_msg->height = height;

            left_cam_info_msg->header.frame_id = left_frame_id;
            right_cam_info_msg->header.frame_id = right_frame_id;
        }

       void callback(zed_wrapper::ZedConfig &config, uint32_t level) {
            NODELET_INFO("Reconfigure: rgbrate %d", config.rgbrate);
            NODELET_INFO("Reconfigure: confidence %d", config.confidence);
            confidence = config.confidence;
        }

        void device_poll() {
            ros::Rate loop_rate(rate);
            ros::Time old_t = ros::Time::now();
            bool old_image = false;

            // Get the parameters of the ZED images
            int width = zed.getResolution().width;
            int height = zed.getResolution().height;
            NODELET_DEBUG_STREAM("Image size : " << width << "x" << height);

            cv::Size cvSize(width, height);
            cv::Mat leftImRGB(cvSize, CV_8UC3);
            cv::Mat rightImRGB(cvSize, CV_8UC3);

            // Create and fill the camera information messages
            sensor_msgs::CameraInfoPtr rgb_cam_info_msg(new sensor_msgs::CameraInfo());
            sensor_msgs::CameraInfoPtr left_cam_info_msg(new sensor_msgs::CameraInfo());
            sensor_msgs::CameraInfoPtr right_cam_info_msg(new sensor_msgs::CameraInfo());
            sensor_msgs::CameraInfoPtr depth_cam_info_msg(new sensor_msgs::CameraInfo());
            fillCamInfo(zed, left_cam_info_msg, right_cam_info_msg, left_frame_id, right_frame_id);
            rgb_cam_info_msg = depth_cam_info_msg = left_cam_info_msg; // the reference camera is the Left one (next to the ZED logo)


            sl::RuntimeParameters runParams;
            runParams.sensing_mode = static_cast<sl::SENSING_MODE> (sensing_mode);

            sl::Mat leftZEDMat, rightZEDMat, depthZEDMat;
            // Main loop
            while (nh_ns.ok()) {
                // Check for subscribers
                int rgb_SubNumber = pub_rgb.getNumSubscribers();
                int rgb_raw_SubNumber = pub_raw_rgb.getNumSubscribers();
                int left_SubNumber = pub_left.getNumSubscribers();
                int left_raw_SubNumber = pub_raw_left.getNumSubscribers();
                int right_SubNumber = pub_right.getNumSubscribers();
                int right_raw_SubNumber = pub_raw_right.getNumSubscribers();
                int depth_SubNumber = pub_depth.getNumSubscribers();
                bool runLoop = (rgb_SubNumber + rgb_raw_SubNumber + left_SubNumber + left_raw_SubNumber + right_SubNumber + right_raw_SubNumber + depth_SubNumber) > 0;

                // Run the loop only if there are some subscribers
                if (runLoop) {

                    computeDepth = depth_SubNumber > 0; // Detect if one of the subscriber need to have the depth information
                    if (computeDepth) {
                        if (!computingDepth) {
                            NODELET_INFO("Starting to compute depth information");
                            computingDepth = true;
                        } 
                    } else {
                        if (computingDepth) {
                            NODELET_INFO("Stopping computing depth information");
                            computingDepth = false;
                        } 
                    }
                    ros::Time t = ros::Time::now(); // Get current time

                    grabbing = true;
                    if (computeDepth) {
                        int actual_confidence = zed.getConfidenceThreshold();
                        if (actual_confidence != confidence)
                            zed.setConfidenceThreshold(confidence);
                        runParams.enable_depth = true; // Ask to compute the depth
                    } else
                        runParams.enable_depth = false;

                    old_image = zed.grab(runParams); // Ask to not compute the depth

                    grabbing = false;
                    if (old_image) { // Detect if a error occurred (for example: the zed have been disconnected) and re-initialize the ZED
                        NODELET_DEBUG("Wait for a new image to proceed");
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        if ((t - old_t).toSec() > 5) {
                            zed.close();

                            NODELET_INFO("Re-opening the ZED");
                            sl::ERROR_CODE err = sl::ERROR_CODE_CAMERA_NOT_DETECTED;
                            while (err != sl::SUCCESS) {
                                err = zed.open(param); // Try to initialize the ZED
                                NODELET_INFO_STREAM(toString(err));
                                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                            }
                        }
                        continue;
                    }

                    old_t = ros::Time::now();

                    // Publish the left == rgb image if someone has subscribed to
                    if (left_SubNumber > 0 || rgb_SubNumber > 0) {
                        // Retrieve RGBA Left image
                        zed.retrieveImage(leftZEDMat, sl::VIEW_LEFT);
                        cv::cvtColor(toCVMat(leftZEDMat), leftImRGB, CV_RGBA2RGB);
                        if (left_SubNumber > 0) {
                            publishCamInfo(left_cam_info_msg, pub_left_cam_info, t);
                            publishImage(leftImRGB, pub_left, left_frame_id, t);
                        }
                        if (rgb_SubNumber > 0) {
                            publishCamInfo(rgb_cam_info_msg, pub_rgb_cam_info, t);
                            publishImage(leftImRGB, pub_rgb, rgb_frame_id, t); // rgb is the left image
                        }
                    }

                    // Publish the left_raw == rgb_raw image if someone has subscribed to
                    if (left_raw_SubNumber > 0 || rgb_raw_SubNumber > 0) {
                        // Retrieve RGBA Left image
                        zed.retrieveImage(leftZEDMat, sl::VIEW_LEFT_UNRECTIFIED);
                        cv::cvtColor(toCVMat(leftZEDMat), leftImRGB, CV_RGBA2RGB);
                        if (left_raw_SubNumber > 0) {
                            publishCamInfo(left_cam_info_msg, pub_left_cam_info, t);
                            publishImage(leftImRGB, pub_raw_left, left_frame_id, t);
                        }
                        if (rgb_raw_SubNumber > 0) {
                            publishCamInfo(rgb_cam_info_msg, pub_rgb_cam_info, t);
                            publishImage(leftImRGB, pub_raw_rgb, rgb_frame_id, t);
                        }
                    }

                    // Publish the right image if someone has subscribed to
                    if (right_SubNumber > 0) {
                        // Retrieve RGBA Right image
                        zed.retrieveImage(rightZEDMat, sl::VIEW_RIGHT);
                        cv::cvtColor(toCVMat(rightZEDMat), rightImRGB, CV_RGBA2RGB);
                        publishCamInfo(right_cam_info_msg, pub_right_cam_info, t);
                        publishImage(rightImRGB, pub_right, right_frame_id, t);
                    }

                    // Publish the right image if someone has subscribed to
                    if (right_raw_SubNumber > 0) {
                        // Retrieve RGBA Right image
                        zed.retrieveImage(rightZEDMat, sl::VIEW_RIGHT_UNRECTIFIED);
                        cv::cvtColor(toCVMat(rightZEDMat), rightImRGB, CV_RGBA2RGB);
                        publishCamInfo(right_cam_info_msg, pub_right_cam_info, t);
                        publishImage(rightImRGB, pub_raw_right, right_frame_id, t);
                    }

                    // Publish the depth image if someone has subscribed to
                    if (depth_SubNumber > 0) {
                        zed.retrieveMeasure(depthZEDMat, sl::MEASURE_DEPTH);
                        publishCamInfo(depth_cam_info_msg, pub_depth_cam_info, t);
                        publishDepth(toCVMat(depthZEDMat), pub_depth, depth_frame_id, t, width, height); // in meters
                    }

                    loop_rate.sleep();
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10)); // No subscribers, we just wait
                }
            } // while loop
            zed.close();
        }

        boost::shared_ptr<dynamic_reconfigure::Server<zed_wrapper::ZedConfig>> server;

        void onInit() {
            // Launch file parameters
            resolution = sl::RESOLUTION_HD720;
            quality = sl::DEPTH_MODE_PERFORMANCE;
            sensing_mode = sl::SENSING_MODE_STANDARD;
            rate = 5;
            gpu_id = -1;
            zed_id = 0;

            nh = getMTNodeHandle();
            nh_ns = getMTPrivateNodeHandle();

            // Get parameters from launch file
            nh_ns.getParam("resolution", resolution);
            nh_ns.getParam("quality", quality);
            nh_ns.getParam("sensing_mode", sensing_mode);
            nh_ns.getParam("frame_rate", rate);
            nh_ns.getParam("gpu_id", gpu_id);
            nh_ns.getParam("zed_id", zed_id);
            nh_ns.getParam("depth_stabilization", depth_stabilization);
            nh_ns.getParam("confidence", confidence);

            std::string img_topic = "image_rect_color";
            std::string img_raw_topic = "image_raw_color";

            // Set the default topic names
            string left_topic = "left/" + img_topic;
            string left_raw_topic = "left/" + img_raw_topic;
            string left_cam_info_topic = "left/camera_info";
            left_frame_id = camera_frame_id;

            string right_topic = "right/" + img_topic;
            string right_raw_topic = "right/" + img_raw_topic;
            string right_cam_info_topic = "right/camera_info";
            right_frame_id = camera_frame_id;

            string rgb_topic = "rgb/" + img_topic;
            string rgb_raw_topic = "rgb/" + img_raw_topic;
            string rgb_cam_info_topic = "rgb/camera_info";
            rgb_frame_id = depth_frame_id;

            string depth_topic = "depth/";
            depth_topic += "depth_registered";

            string depth_cam_info_topic = "depth/camera_info";

            camera_to_vr.setIdentity();  // set transform to identity till we get a valid tf

            nh_ns.getParam("rgb_topic", rgb_topic);
            nh_ns.getParam("rgb_raw_topic", rgb_raw_topic);
            nh_ns.getParam("rgb_cam_info_topic", rgb_cam_info_topic);

            nh_ns.getParam("left_topic", left_topic);
            nh_ns.getParam("left_raw_topic", left_raw_topic);
            nh_ns.getParam("left_cam_info_topic", left_cam_info_topic);

            nh_ns.getParam("right_topic", right_topic);
            nh_ns.getParam("right_raw_topic", right_raw_topic);
            nh_ns.getParam("right_cam_info_topic", right_cam_info_topic);

            nh_ns.getParam("depth_topic", depth_topic);
            nh_ns.getParam("depth_cam_info_topic", depth_cam_info_topic);

            // Initialization transformation listener
            tfBuffer.reset( new tf2_ros::Buffer );
            tf_listener.reset( new tf2_ros::TransformListener(*tfBuffer) );

            // Try to initialize the ZED

            if (rate >= 15) {
                param.camera_fps = rate;
            } else {
                param.camera_fps = 15;  // camera will only set to min 15
            }
            param.camera_resolution = static_cast<sl::RESOLUTION> (resolution);
            param.camera_buffer_count_linux = 1;  // to cut latency

            param.camera_linux_id = zed_id;

            param.coordinate_units = sl::UNIT_METER;
            param.coordinate_system = sl::COORDINATE_SYSTEM_IMAGE;
            param.depth_mode = static_cast<sl::DEPTH_MODE> (quality);
            param.sdk_verbose = true;
            param.sdk_gpu_id = gpu_id;
            param.depth_stabilization = depth_stabilization;

            // SVRT changes for better point clouds
            // depth between .3 and 5 meters
            param.depth_minimum_distance = .3;

            sl::ERROR_CODE err = sl::ERROR_CODE_CAMERA_NOT_DETECTED;
            while (err != sl::SUCCESS) {
                err = zed.open(param);
                NODELET_INFO_STREAM(toString(err));
                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            }

            NODELET_INFO_STREAM("SVRT parameters set");
            // SVRT changes for better point clouds

            // disable tracking
            zed.disableTracking();

            // depth between .3 and 5 meters
            zed.setDepthMaxRangeValue(5.0);

            // first set camera parameters to default
            zed.setCameraSettings(sl::CAMERA_SETTINGS_BRIGHTNESS, -1, true);
            zed.setCameraSettings(sl::CAMERA_SETTINGS_CONTRAST, -1, true);
            zed.setCameraSettings(sl::CAMERA_SETTINGS_HUE, -1, true);
            zed.setCameraSettings(sl::CAMERA_SETTINGS_SATURATION, -1, true);
            zed.setCameraSettings(sl::CAMERA_SETTINGS_GAIN, -1, true);
            zed.setCameraSettings(sl::CAMERA_SETTINGS_EXPOSURE, -1, true);
            zed.setCameraSettings(sl::CAMERA_SETTINGS_WHITEBALANCE, -1, true);

            // now adjust contrast, saturation, gain to get better images
            // 0 to 8
            //zed.setCameraSettings(sl::CAMERA_SETTINGS_BRIGHTNESS, -1);
            // 0 to 8
            zed.setCameraSettings(sl::CAMERA_SETTINGS_CONTRAST, 8);
            // 0 to 11
            //zed.setCameraSettings(sl::CAMERA_SETTINGS_HUE, -1);
            // 0 to 8
            //zed.setCameraSettings(sl::CAMERA_SETTINGS_SATURATION, 8);
            // 0 to 100, or auto if EXPOSURE is -1
            //zed.setCameraSettings(sl::CAMERA_SETTINGS_GAIN, -1, true);
            // 0 to 100, -1 AutoExposure/AutoGain
            //zed.setCameraSettings(sl::CAMERA_SETTINGS_EXPOSURE, -1, true);
            // 2800 to 6500 by 100, -1 is Auto White Balance
            //zed.setCameraSettings(sl::CAMERA_SETTINGS_WHITEBALANCE, -1, true);

            //Reconfigure for various parameters
            server = boost::make_shared<dynamic_reconfigure::Server<zed_wrapper::ZedConfig>>();
            dynamic_reconfigure::Server<zed_wrapper::ZedConfig>::CallbackType f;
            f = boost::bind(&ZEDWrapperNodelet::callback, this, _1, _2);
            server->setCallback(f);

            // Create all the publishers
            // Image publishers
            image_transport::ImageTransport it_zed(nh);
            pub_rgb = it_zed.advertise(rgb_topic, 1); //rgb
            NODELET_INFO_STREAM("Advertized on topic " << rgb_topic);
            pub_raw_rgb = it_zed.advertise(rgb_raw_topic, 1); //rgb raw
            NODELET_INFO_STREAM("Advertized on topic " << rgb_raw_topic);
            pub_left = it_zed.advertise(left_topic, 1); //left
            NODELET_INFO_STREAM("Advertized on topic " << left_topic);
            pub_raw_left = it_zed.advertise(left_raw_topic, 1); //left raw
            NODELET_INFO_STREAM("Advertized on topic " << left_raw_topic);
            pub_right = it_zed.advertise(right_topic, 1); //right
            NODELET_INFO_STREAM("Advertized on topic " << right_topic);
            pub_raw_right = it_zed.advertise(right_raw_topic, 1); //right raw
            NODELET_INFO_STREAM("Advertized on topic " << right_raw_topic);
            pub_depth = nh.advertise<rodan_vr_api::CompressedDepth>(depth_topic, 1); //depth
            NODELET_INFO_STREAM("Advertized on topic " << depth_topic);

            // Camera info publishers
            pub_rgb_cam_info = nh.advertise<sensor_msgs::CameraInfo>(rgb_cam_info_topic, 1); //rgb
            NODELET_INFO_STREAM("Advertized on topic " << rgb_cam_info_topic);
            pub_left_cam_info = nh.advertise<sensor_msgs::CameraInfo>(left_cam_info_topic, 1); //left
            NODELET_INFO_STREAM("Advertized on topic " << left_cam_info_topic);
            pub_right_cam_info = nh.advertise<sensor_msgs::CameraInfo>(right_cam_info_topic, 1); //right
            NODELET_INFO_STREAM("Advertized on topic " << right_cam_info_topic);
            pub_depth_cam_info = nh.advertise<sensor_msgs::CameraInfo>(depth_cam_info_topic, 1); //depth
            NODELET_INFO_STREAM("Advertized on topic " << depth_cam_info_topic);

            NODELET_INFO_STREAM("Publish rate " << rate << " Hz");
            NODELET_INFO_STREAM("GPU count " << getCudaEnabledDeviceCount());

            device_poll_thread = boost::shared_ptr<boost::thread>
                    (new boost::thread(boost::bind(&ZEDWrapperNodelet::device_poll, this)));
        }
    }; // class ZEDROSWrapperNodelet
} // namespace

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(zed_wrapper::ZEDWrapperNodelet, nodelet::Nodelet);
