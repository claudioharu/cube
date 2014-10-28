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

// functions 
static void DrawOpencvImage( Mat* image );
static void Redraw (void*);



int height, width;
double fovx,fovy,focalLength,aspectRatio;

Point2d principalPt;

Mat img_scene;
Mat rvec = Mat(Size(3,1), CV_64F);
Mat tvec = Mat(Size(3,1), CV_64F);
Mat intrinsics, distortion;

// video with opengl cube
VideoWriter opengl;


int main( int argc, char** argv )
{

    FileStorage fs;
    fs.open("calibracao.yml", FileStorage::READ);

    fs["intrinsic"] >> intrinsics;
    fs["distortion"] >> distortion;

    Mat src;
    Mat target = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );


    // sharpening scene
    GaussianBlur(target, src, Size(0, 0), 3);
    addWeighted(target, 1.5, src, -0.5, 0, target);

    Mat img_scene_gray;

    // get a new frame
    VideoCapture cap("cena.avi"); 

    // checking     
    if(!cap.isOpened())
    {
       printf("error\n");
       return -1;
    } 

    // video with opencv cube
    VideoWriter outputVideo;

    // getting information from the input video
    Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    height = (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    width = (int) cap.get(CV_CAP_PROP_FRAME_WIDTH);

    // output without opengl cube
    // outputVideo.open("output.avi" , ex, cap.get(CV_CAP_PROP_FPS),S, true);

    // getting fovy, fovx, localLenght... 
    calibrationMatrixValues(intrinsics, S, 0.0, 0.0, fovx, fovy, focalLength, principalPt, aspectRatio);

    // opngl video
    opengl.open("outOpengl.avi" , ex, cap.get(CV_CAP_PROP_FPS),S, true );

    // window opencv
    namedWindow("window", CV_WINDOW_OPENGL);
    setOpenGlContext("window");
    setOpenGlDrawCallback("window", Redraw);
    resizeWindow("window", width, height);

    // openGL functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
    glutInitWindowSize( width, height );

    int minHessian = 400;
    
    while(true)
    {
        // get a new frame
        bool success = cap.read(img_scene); 
        if(!success)
            break;
     
        // sharpening scene
        GaussianBlur(img_scene, src, Size(0, 0), 3);
        addWeighted(img_scene, 1.5, src, -0.5, 0, img_scene);

        // converting color
        cvtColor(img_scene, img_scene_gray, CV_BGR2GRAY);

        // using surf detector  
        SurfFeatureDetector detect( minHessian );

        std::vector<KeyPoint> keypoints_object;
        std::vector<KeyPoint> keypoints_scene;

        // detecting keyPoints
        detect.detect( target, keypoints_object );
        detect.detect( img_scene_gray, keypoints_scene );

        SurfDescriptorExtractor ext;

        Mat desc_object;
        Mat desc_scene;

        // getting descriptions
        ext.compute( target, keypoints_object, desc_object );
        ext.compute( img_scene_gray, keypoints_scene, desc_scene );

        FlannBasedMatcher match;
        std::vector< DMatch > matches;

        // matching
        match.match( desc_object, desc_scene, matches );

        double max_dist = 0; 
        double min_dist = 100;

        for( int i = 0; i < desc_object.rows; i++ )
        { 
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

    
        std::vector< DMatch > good_matches;

        // getting good matches
        for( int i = 0; i < desc_object.rows; i++ )
        { 
            if( matches[i].distance < max(4*min_dist, 0.02) )
            { 
                good_matches.push_back( matches[i]); 
            }
        }

        std::vector<Point2f> obj;
        std::vector<Point2f> scene;

        // getting the keypoints from the good matches
        for( int i = 0; i < good_matches.size(); i++ )
        {
            obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        }


        if(good_matches.size() >= 4)
        {

            // getting homography
            Mat H = findHomography( obj, scene, CV_RANSAC );

            std::vector<Point2f> objCorners(4);

            // target corners   
            objCorners[0] = cvPoint(0,0); 
            objCorners[1] = cvPoint( target.cols, 0 );
            objCorners[2] = cvPoint( target.cols, target.rows ); 
            objCorners[3] = cvPoint( 0, target.rows );


            vector<Point2f> sceneCorners(4);

            perspectiveTransform( objCorners, sceneCorners, H);

            vector<Point3d> framePoints;

            // cube points
            framePoints.push_back( Point3d( 0.0, 0.0, 0.0 ) ); //0
            framePoints.push_back( Point3d( 0.0, 0.0, -50 ) ); //1
            framePoints.push_back( Point3d( 50, 0.0, 0.0 ) ); //2
            framePoints.push_back( Point3d( 50, 0.0, -50.0 ) ); //3
            framePoints.push_back( Point3d( 0, 50.0, 0.0 ) ); //4
            framePoints.push_back( Point3d( 0.0, 50.0, -50.0 ) ); //5
            framePoints.push_back( Point3d( 50.0, 50.0, 0.0 ) ); //6
            framePoints.push_back( Point3d( 50.0, 50.0, -50.0 ) ); //7

            vector<Point2f> imgPoints;
            vector<Point3f> objPoints;
            for(int i = 0; i < 4; i++)
            {
                imgPoints.push_back(sceneCorners[i]);
                objPoints.push_back(Point3f(objCorners[i].x - target.cols/2, objCorners[i].y - target.rows/2, 0));
            }

            // finding the camera extrinsic parameters
            solvePnP(objPoints, imgPoints, intrinsics, distortion, rvec, tvec);

            vector<Point2d> imgFrameP;

            // projecting 
            projectPoints(framePoints, rvec, tvec, intrinsics, distortion, imgFrameP );


            // drawing the cube
            line(img_scene, imgFrameP[0], imgFrameP[1], CV_RGB(0,0,255), 4 );
            line(img_scene, imgFrameP[0], imgFrameP[2], CV_RGB(0,0,255), 4 );
            line(img_scene, imgFrameP[2], imgFrameP[3], CV_RGB(0,0,255), 4 );
            line(img_scene, imgFrameP[2], imgFrameP[6], CV_RGB(0,0,255), 4 );
            line(img_scene, imgFrameP[3], imgFrameP[7], CV_RGB(0,0,255), 4 );
            line(img_scene, imgFrameP[4], imgFrameP[6], CV_RGB(0,0,255), 4 );
            line(img_scene, imgFrameP[1], imgFrameP[3], CV_RGB(0,0,255), 4 );
            line(img_scene, imgFrameP[0], imgFrameP[4], CV_RGB(0,0,255), 4 );
            line(img_scene, imgFrameP[1], imgFrameP[5], CV_RGB(0,0,255), 4 );
            line(img_scene, imgFrameP[5], imgFrameP[7], CV_RGB(0,0,255), 4 );
            line(img_scene, imgFrameP[6], imgFrameP[7], CV_RGB(0,0,255), 4 );
            line(img_scene, imgFrameP[4], imgFrameP[5], CV_RGB(0,0,255), 4 );

            // writting opencv's video 
            // outputVideo.write(img_scene); 
          }
          else
          {
            // writting opencv's video
            // outputVideo.write( img_scene);
          }

        updateWindow("window");

        if(waitKey(30) >= 0) break;
    }

    setOpenGlDrawCallback("window", 0, 0);
    destroyAllWindows();

    waitKey(0);
    return 0;
}

// DrawOpencvImage( Mat* image )
// This function draws the scene
// Parameters: frame of the scene
static void DrawOpencvImage( Mat* image )
{
    Mat aux;
    flip(*image,aux,0);

    // drawing the scene
    glDrawPixels(width,height,GL_BGR_EXT,GL_UNSIGNED_BYTE,aux.data);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// Redraw (void*)
// This function updates the window
// It draws the cube in every frame
static void Redraw (void*)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0,width, 0.0, height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // drawing the scene
    DrawOpencvImage( &img_scene );


    // light's parameters
    float pos[4] = { 0.0f, 10.0f, 0.0f, 0.0f };
    glLightfv( GL_LIGHT0, GL_POSITION, pos );
    float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float red[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
    glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, red );
    glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, white );
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 50.0f );

    glMatrixMode(GL_PROJECTION);

    gluPerspective( fovy, 1.0/aspectRatio, 0.1, 1000.0);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glLoadIdentity();

    glScaled(0.5,0.5,0.5);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // rotating model
    glRotatef(rvec.at<double>(0)*30, 1,0,0);
    glRotatef(rvec.at<double>(1)*(-30), 0,1,0);
    glRotatef(rvec.at<double>(2)*(-30), 0,0,1);

    glScaled(0.1,0.1,1);

    // translating model
    glTranslatef(tvec.at<double>(0)/22.0f,-tvec.at<double>(1)/22.0f, 0);

    // glutWireCube(1.0); 
    glutSolidCube(1.0);


    // writing the output video
    Mat aux = Mat(height, width, CV_8UC3);
    glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, aux.data);
    flip(aux,aux,0);
    // writting opengl's video
    opengl.write(aux);
    aux.release();
}