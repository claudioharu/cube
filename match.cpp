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

/** @function main */
void displayMe(void*);
static void Redraw (void*);
// double fovx,fovy,focalLength,aspectRatio; Point2d principalPt;
VideoWriter opengl;


Mat img_scene;
Mat rvec = Mat(Size(3,1), CV_64F);
Mat tvec = Mat(Size(3,1), CV_64F);
int height, width;
Mat rotation; 
Mat intrinsics, distortion;

Mat viewMatrix(4, 4, CV_64F); 


int main( int argc, char** argv )
{

    FileStorage fs;
    fs.open("calibracao.yml", FileStorage::READ);


    fs["intrinsic"] >> intrinsics;
    fs["distortion"] >> distortion;

    std::cout << intrinsics << "\n";
    // std::cout << distortion << "\n";


    // calibrationMatrixValues(intrinsics, Size(640,480), 0.0, 0.0, fovx, fovy, focalLength, principalPt, aspectRatio);

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

    // ogl::Texture2D tex;

    // tex.create(S, cv::ogl::Texture2D::RGB, false);

    opengl.open("outOpengl.avi" , ex, cap.get(CV_CAP_PROP_FPS),S, true );

    setOpenGlContext("window");

    
    setOpenGlDrawCallback("window", Redraw);

    height = (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    width = (int) cap.get(CV_CAP_PROP_FRAME_WIDTH);

    resizeWindow("window", width, height);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
    glutInitWindowSize( width, height  );
    // glutCreateWindow("Realidade Aumentada");
    
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


            Rodrigues(rvec, rotation);

            viewMatrix = Mat::zeros(4, 4, CV_64FC1);

            for(int row = 0; row < 3; row++)
            {
               for(int col = 0; col < 3; col++)
               {
                  viewMatrix.at<double>(row, col) = rotation.at<double>(row, col);
               }
               viewMatrix.at<double>(row, 3) = tvec.at<double>(row, 0);
            }
            viewMatrix.at<double>(3, 3) = 1.0;


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



            outputVideo.write(img_scene); 
          }
          else
          {

            outputVideo.write( img_scene);
          }

        updateWindow("window");

        if(waitKey(30) >= 0) break;
    }

    setOpenGlDrawCallback("window", 0, 0);
    destroyAllWindows();

  waitKey(0);
  return 0;
}


static void DrawOpencvImage( Mat* image )
{
    Mat aux;
    flip(*image,aux,0);
    glDrawPixels(width,height,GL_BGR_EXT,GL_UNSIGNED_BYTE,aux.data);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

static void Redraw (void*)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0,width, 0.0, height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    DrawOpencvImage( &img_scene );

    Mat cvToGl = Mat::eye(4, 4, CV_64F); 
    cvToGl.at<double>(1, 1) = -1; // Invert the y axis 
    cvToGl.at<double>(2, 2) = -1; // invert the z axis 
    viewMatrix = cvToGl * viewMatrix;
    
    // std::cout << viewMatrix << "\n";
    GLdouble pr[16], mv[16];
    double fx,fy,cx,cy;


    fx = intrinsics.at<double>(0, 0);
    fy = intrinsics.at<double>(1, 1);
    cx = intrinsics.at<double>(0, 2);
    cy = intrinsics.at<double>(1, 2);

    double np = 0.39;
    double fp = 100.0;


    pr[0] = 2 * fx / width;
    pr[1] = 0;
    pr[2] = 0;
    pr[3] = 0;

    pr[4] = 0;
    pr[5] = 2 * fy / height;
    pr[6] = 0;
    pr[7] = 0;

    pr[8]  = (2 * cx / width) - 1;
    pr[9]  = (2 * cy / height) - 1;
    pr[10] = (fp+np)/(fp-np);
    pr[11] =  1;

    pr[12] = 0;
    pr[13] = 0;
    pr[14] = -2*fp*np/(fp-np);
    pr[15] = 0;


    mv[0]  = viewMatrix.at<double>(0, 0);
    mv[1]  = viewMatrix.at<double>(1, 0);
    mv[2]  = viewMatrix.at<double>(2, 0);
    mv[3]  = 0;

    mv[4]  = viewMatrix.at<double>(0, 1);
    mv[5]  = viewMatrix.at<double>(1, 1);
    mv[6]  = viewMatrix.at<double>(2, 1);
    mv[7]  = 0;

    mv[8]  = -viewMatrix.at<double>(0, 2);
    mv[9]  = -viewMatrix.at<double>(1, 2);
    mv[10] = -viewMatrix.at<double>(2, 2);
    mv[11] = 0;

    mv[12] = 0;
    mv[13] = 0;
    mv[14] = 0;
    mv[15] = 1;


    float pos[4] = { 0.0f, 10.0f, 0.0f, 0.0f };
    glLightfv( GL_LIGHT0, GL_POSITION, pos );

    float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float red[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
    glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, red );
    glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, white );
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 50.0f );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // // glLoadMatrixd(mtsaiopengl_projection);
    // glLoadMatrixd(m);

    glLoadMatrixd(pr);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);


    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glLoadMatrixd(mv);
  
    glPushMatrix();  

    glScaled(0.1,0.1,1);
    glScaled(0.5,0.5,1);
    glTranslatef(0,1,0);
  
    glRotatef(90, 1.0, 0.0, 0.0);
    // glutSolidTeapot(1.0);
    glutSolidCube(1.0);
    glPopMatrix();


    Mat aux = Mat(height, width, CV_8UC3);
    glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, aux.data);
    flip(aux,aux,0);

    opengl.write(aux);
    aux.release();
    // glutSwapBuffers();
}