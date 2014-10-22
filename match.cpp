#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>


#include <GL/gl.h>
#include <GL/glut.h>
#include <opencv2/core/opengl_interop.hpp>


using namespace cv;

void displayMe(void*);

/** @function main */
Mat img_scene;
    Mat rvec = Mat(Size(3,1), CV_64F);
    Mat tvec = Mat(Size(3,1), CV_64F);

int main( int argc, char** argv )
{

    FileStorage fs;
    fs.open("calibracao.yml", FileStorage::READ);

    Mat intrinsics, distortion;

    fs["intrinsic"] >> intrinsics;
    fs["distortion"] >> distortion;

    // std::cout << intrinsics << "\n";
    // std::cout << distortion << "\n";



    Mat src;
    Mat img_object = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );

    GaussianBlur(img_object, src, Size(0, 0), 3);
    addWeighted(img_object, 1.5, src, -0.5, 0, img_object);


    Mat img_scene_gray;

    VideoCapture cap("cena.avi"); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
    {
       printf("leu n\n");
       return -1;
    } 

    VideoWriter outputVideo;
    Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    
    int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    outputVideo.open("output.avi" , ex, cap.get(CV_CAP_PROP_FPS),S, true);

    namedWindow("window", CV_WINDOW_OPENGL);
    

    ogl::Texture2D tex;

    tex.create(S, cv::ogl::Texture2D::RGB, false);

    setOpenGlContext("window");

    
    setOpenGlDrawCallback("window", displayMe, &tex);

    

    while(true)
    {
        //cap >> frame; // get a new frame from camera
        bool success = cap.read(img_scene); 
        if(!success)
            break;

            
        // Sharpening
        GaussianBlur(img_scene, src, Size(0, 0), 3);
        addWeighted(img_scene, 1.5, src, -0.5, 0, img_scene);



        cvtColor(img_scene, img_scene_gray, CV_BGR2GRAY);
        //Do your processing here
        //...
      
        //-- Step 1: Detect the keypoints using SURF Detector
        int minHessian = 400;

        SurfFeatureDetector detector( minHessian );

        std::vector<KeyPoint> keypoints_object, keypoints_scene;

        detector.detect( img_object, keypoints_object );
        detector.detect( img_scene_gray, keypoints_scene );

        //-- Step 2: Calculate descriptors (feature vectors)
        SurfDescriptorExtractor extractor;

        Mat descriptors_object, descriptors_scene;

        extractor.compute( img_object, keypoints_object, descriptors_object );
        extractor.compute( img_scene_gray, keypoints_scene, descriptors_scene );



        //-- Step 3: Matching descriptor vectors using FLANN matcher
        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;
        matcher.match( descriptors_object, descriptors_scene, matches );

        double max_dist = 0; double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_object.rows; i++ )
        { 
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

          // printf("-- Max dist : %f \n", max_dist );
          // printf("-- Min dist : %f \n", min_dist );

        //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
        std::vector< DMatch > good_matches;

        for( int i = 0; i < descriptors_object.rows; i++ )
        { 
            if( matches[i].distance < max(4*min_dist, 0.02) )
            { 
                good_matches.push_back( matches[i]); 
            }
        }

        //-- Localize the object
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;

        for( int i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        }


        if(good_matches.size() >= 4)
        {
            Mat H = findHomography( obj, scene, CV_RANSAC );

            //-- Get the corners from the image_1 ( the object to be "detected" )
            std::vector<Point2f> obj_corners(4);
            obj_corners[0] = cvPoint(0,0); 
            obj_corners[1] = cvPoint( img_object.cols, 0 );
            obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); 
            obj_corners[3] = cvPoint( 0, img_object.rows );
            vector<Point2f> scene_corners(4);

            perspectiveTransform( obj_corners, scene_corners, H);

            vector<Point3d> framePoints;
            framePoints.push_back( Point3d( 0.0, 0.0, 0.0 ) ); //0
            framePoints.push_back( Point3d( 0.0, 0.0, -50 ) ); //1

            framePoints.push_back( Point3d( 50, 0.0, 0.0 ) ); //2
            framePoints.push_back( Point3d( 50, 0.0, -50.0 ) ); //3

            framePoints.push_back( Point3d( 0, 50.0, 0.0 ) ); //4
            framePoints.push_back( Point3d( 0.0, 50.0, -50.0 ) ); //5

            framePoints.push_back( Point3d( 50.0, 50.0, 0.0 ) ); //6
            framePoints.push_back( Point3d( 50.0, 50.0, -50.0 ) ); //7



            //SHOW ROTATION AND TRANSLATION VECTORS
            vector<Point2f> image_points;
            vector<Point3f> object_points;
            for(int i = 0; i < 4; i++)
            {
                image_points.push_back(scene_corners[i]);
                object_points.push_back(Point3f(obj_corners[i].x-img_object.cols/2, obj_corners[i].y-img_object.rows/2, 0));
            }

            //find the camera extrinsic parameters
            solvePnP(object_points, image_points, intrinsics, distortion, rvec, tvec);

            vector<Point2d> imageFramePoints;
            projectPoints(framePoints, rvec, tvec, intrinsics, distortion, imageFramePoints );


            // -- Draw lines between the corners (the mapped object in the scene - image_2 )
            line( img_scene, scene_corners[0] , scene_corners[1] , Scalar(0, 255, 0), 4 );
            line( img_scene, scene_corners[1] , scene_corners[2] , Scalar( 0, 255, 0), 4 );
            line( img_scene, scene_corners[2] , scene_corners[3] , Scalar( 0, 255, 0), 4 );
            line( img_scene, scene_corners[3] , scene_corners[0], Scalar( 0, 255, 0), 4 );



            line(img_scene, imageFramePoints[0], imageFramePoints[1], CV_RGB(0,0,255), 4 );
            line(img_scene, imageFramePoints[0], imageFramePoints[2], CV_RGB(0,0,255), 4 );
            line(img_scene, imageFramePoints[2], imageFramePoints[3], CV_RGB(0,0,255), 4 );
            line(img_scene, imageFramePoints[1], imageFramePoints[3], CV_RGB(0,0,255), 4 );
            line(img_scene, imageFramePoints[0], imageFramePoints[4], CV_RGB(0,0,255), 4 );
            line(img_scene, imageFramePoints[1], imageFramePoints[5], CV_RGB(0,0,255), 4 );
            line(img_scene, imageFramePoints[2], imageFramePoints[6], CV_RGB(0,0,255), 4 );
            line(img_scene, imageFramePoints[3], imageFramePoints[7], CV_RGB(0,0,255), 4 );
            line(img_scene, imageFramePoints[4], imageFramePoints[6], CV_RGB(0,0,255), 4 );
            line(img_scene, imageFramePoints[6], imageFramePoints[7], CV_RGB(0,0,255), 4 );
            line(img_scene, imageFramePoints[4], imageFramePoints[5], CV_RGB(0,0,255), 4 );
            line(img_scene, imageFramePoints[5], imageFramePoints[7], CV_RGB(0,0,255), 4 );



            // std::cout << imageFramePoints[0];

            // line(img_scene, imageFramePoints[1], imageFramePoints[2], CV_RGB(255,255,0), 4 );
            // line(img_scene, imageFramePoints[1], imageFramePoints[3], CV_RGB(0,255,255), 4 );
            // line(img_scene, imageFramePoints[2], imageFramePoints[3], CV_RGB(0,0,0), 4 );

            tex.copyFrom(img_scene);

            //-- Show detected matches
            
            // imshow( "window", img_scene );
            // setOpenGlDrawCallback("window", displayMe);
            outputVideo.write(img_scene); 
          }
          else
          {
            tex.copyFrom(img_scene);
             // imshow( "window", img_scene); 
            outputVideo.write( img_scene);
          }

        updateWindow("window");
        // std::cout << s;
        //Show the image
        // imshow("Output", frame);
        //if(waitKey(0) == 27) break;
        if(waitKey(30) >= 0) break;
    }

    setOpenGlDrawCallback("window", 0, 0);
    destroyAllWindows();

  // if( !img_object.data || !img_scene.data )
  // { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  
  waitKey(0);
  return 0;
}




void displayMe(void* userdata)
{
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    ogl::Texture2D* pTex = static_cast<ogl::Texture2D*>(userdata);
    if (pTex->empty())
        return;

    glLoadIdentity();



    glEnable(GL_TEXTURE_2D);
    pTex->bind();

    Rect_<double> wndRect=Rect_<double>(0.0, 0.0, 1.0, 1.0);
    Rect_<double> texRect=Rect_<double>(0.0, 0.0, 1.0, 1.0);


    ogl::render(*pTex, wndRect, texRect);

    pTex->release();

    // Reset transformations
    glLoadIdentity();

    glScaled(0.5,0.5,0.5);
    
    // glMatrixMode(GL_PROJECTION);

    glRotatef(rvec.at<double>(0)*10, 1,0,0);
    glRotatef(rvec.at<double>(1)*10, 0,1,0);
    glRotatef(rvec.at<double>(2)*10, 0,0,1);

    glScaled(0.5,0.5,0.5);
    glScaled(0.5,0.5,0.5);
    glScaled(0.5,0.5,0.5);
    glScaled(0.5,0.5,0.5);

    // glTranslatef(10.5f, 0.0f, 0.0f);  // Move right and into the screen
    // glTranslatef(tvec.at<double>(0),0,0);//tvec.at<double>(1)-129,tvec.at<double>(2)/50);
    // glTranslatef(0,tvec.at<double>(1),0);

    // glTranslatef(tvec.at<double>(0)-10,tvec.at<double>(1)-129,tvec.at<double>(2)/50);

    // std::cout << tvec;

    glBegin(GL_QUADS);  
      glColor3f(0.0f, 1.0f, 0.0f); // Green

      glVertex3f( 1.0f, 1.0f, 0.0f);
      glVertex3f(-1.0f, 1.0f, 0.0f);
      glVertex3f(-1.0f, 1.0f,  1.0f);
      glVertex3f( 1.0f, 1.0f,  1.0f);
 
      // Bottom face (y = -1.0f)
      glColor3f(1.0f, 0.5f, 0.0f);     // Orange
      glVertex3f( 1.0f, -1.0f,  1.0f);
      glVertex3f(-1.0f, -1.0f,  1.0f);
      glVertex3f(-1.0f, -1.0f, 0.0f);
      glVertex3f( 1.0f, -1.0f, 0.0f);
 
      // Front face  (z = 1.0f)
      glColor3f(1.0f, 0.0f, 0.0f);     // Red
      glVertex3f( 1.0f,  1.0f, 1.0f);
      glVertex3f(-1.0f,  1.0f, 1.0f);
      glVertex3f(-1.0f, -1.0f, 1.0f);
      glVertex3f( 1.0f, -1.0f, 1.0f);
 
      // Back face (z = -1.0f)
      glColor3f(1.0f, 1.0f, 0.0f);     // Yellow
      glVertex3f( 1.0f, -1.0f, 0.0f);
      glVertex3f(-1.0f, -1.0f, 0.0f);
      glVertex3f(-1.0f,  1.0f, 0.0f);
      glVertex3f( 1.0f,  1.0f, 0.0f);
 
      // Left face (x = -1.0f)
      glColor3f(0.0f, 0.0f, 1.0f);     // Blue
      glVertex3f(-1.0f,  1.0f,  1.0f);
      glVertex3f(-1.0f,  1.0f, 0.0f);
      glVertex3f(-1.0f, -1.0f, 0.0f);
      glVertex3f(-1.0f, -1.0f,  1.0f);
 
      // Right face (x = 1.0f)
      glColor3f(1.0f, 0.0f, 1.0f);     // Magenta
      glVertex3f(1.0f,  1.0f, 0.0f);
      glVertex3f(1.0f,  1.0f,  1.0f);
      glVertex3f(1.0f, -1.0f,  1.0f);
      glVertex3f(1.0f, -1.0f, 0.0f);
   glEnd();  // End of drawing color-cube


    //  // White side - BACK
    // glBegin(GL_POLYGON);
    // glColor3f(   1.0,  1.0, 1.0 );
    // glVertex3f(  0.5, -0.5, 0.5 );
    // glVertex3f(  0.5,  0.5, 0.5 );
    // glVertex3f( -0.5,  0.5, 0.5 );
    // glVertex3f( -0.5, -0.5, 0.5 );
    // glEnd();
     
    // // Purple side - RIGHT
    // glBegin(GL_POLYGON);
    // glColor3f(  1.0,  0.0,  1.0 );
    // glVertex3f( 0.5, -0.5, -0.5 );
    // glVertex3f( 0.5,  0.5, -0.5 );
    // glVertex3f( 0.5,  0.5,  0.5 );
    // glVertex3f( 0.5, -0.5,  0.5 );
    // glEnd();
     
    // // Green side - LEFT
    // glBegin(GL_POLYGON);
    // glColor3f(   0.0,  1.0,  0.0 );
    // glVertex3f( -0.5, -0.5,  0.5 );
    // glVertex3f( -0.5,  0.5,  0.5 );
    // glVertex3f( -0.5,  0.5, -0.5 );
    // glVertex3f( -0.5, -0.5, -0.5 );
    // glEnd();
     
    // // Blue side - TOP
    // glBegin(GL_POLYGON);
    // glColor3f(   0.0,  0.0,  1.0 );
    // glVertex3f(  0.5,  0.5,  0.5 );
    // glVertex3f(  0.5,  0.5, -0.5 );
    // glVertex3f( -0.5,  0.5, -0.5 );
    // glVertex3f( -0.5,  0.5,  0.5 );
    // glEnd();
     
    // // Red side - BOTTOM
    // glBegin(GL_POLYGON);
    // glColor3f(   1.0,  0.0,  0.0 );
    // glVertex3f(  0.5, -0.5, -0.5 );
    // glVertex3f(  0.5, -0.5,  0.5 );
    // glVertex3f( -0.5, -0.5,  0.5 );
    // glVertex3f( -0.5, -0.5, -0.5 );
    // glEnd();
     
    // glFlush();

}