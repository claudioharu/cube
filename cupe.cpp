// Includes
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#define GL_GLEXT_PROTOTYPES
#include <GLUT/glut.h>
#include <GL/glut.h>

// Function Prototypes
void display();
void specialKeys();
// Global Variables
double rotate_y=0; 
double rotate_x=0

int main(int argc, char* argv[]){
 
	//  Initialize GLUT and process user parameters
	glutInit(&argc,argv);
	 
	//  Request double buffered true color window with Z-buffer
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	
	// Create window
	glutCreateWindow("Awesome Cube");

	//  Enable Z-buffer depth test
	glEnable(GL_DEPTH_TEST);
	// Callback functions
	glutDisplayFunc(display);
	glutSpecialFunc(specialKeys);
	//  Pass control to GLUT for events
	glutMainLoop();
	 
	//  Return to OS
	return 0;
	 
}

void display(){
 
	//  Clear screen and Z-buffer
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	// Multi-colored side - FRONT
	glBegin(GL_POLYGON);
	 
	glVertex3f( -0.5, -0.5, -0.5);       // P1
	glVertex3f( -0.5,  0.5, -0.5);       // P2
	glVertex3f(  0.5,  0.5, -0.5);       // P3
	glVertex3f(  0.5, -0.5, -0.5);       // P4
	// Vertices will be added in the next step
	 
	glEnd();