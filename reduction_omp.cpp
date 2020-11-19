#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <math.h> 
#include <sys/time.h>
#include <cstdlib>
#include <stdlib.h>
#include <omp.h>

using namespace cv;
const int height = 480; 
const int width  = 854;
int n_threads; 
uint8_t * resized; 


struct args{
    int id; 
    Mat * img; 
};

void bilinear_resize(void * input ){ //Mat img, uint8_t * resized){

    int id = ((struct args*)input)->id;
    Mat * img = ((struct args*)input)->img;

    int divi = height/n_threads;
    int row = (id * divi);

    //printf("Number of rows: %d", img.rows );

    uint8_t * pixelPtr = (uint8_t*)img->data;
    int cn = img->channels();

    float x_ratio = (img->cols - 1)/(width - 1) ; 
    float y_ratio = (img->rows - 1)/(height - 1);  

    uint8_t a, b, c, d, pixel;

    for(int i = row; i < min(row+divi, height); i++){
        for(int j = 0; j < width;  j++){

            int x_l = floor(x_ratio * j), y_l = floor(y_ratio * i); 
            int x_h = ceil(x_ratio * j), y_h = ceil(y_ratio * i);

            float x_weight = (x_ratio * j) - x_l;
            float y_weight = (y_ratio * i) - y_l;

            for(int k = 0; k < cn; k++){
                a = pixelPtr[y_l*img->cols*cn + x_l*cn + k];
                b = pixelPtr[y_l*img->cols*cn + x_h*cn + k];
                c = pixelPtr[y_h*img->cols*cn + x_l*cn + k];
                d = pixelPtr[y_h*img->cols*cn + x_h*cn + k];

                pixel =( a&0xff )* (1 - x_weight) * (1 - y_weight)  + (b&0xff) * x_weight * (1 - y_weight) +  (c&0xff) * y_weight * (1 - x_weight) +  (d&0xff ) * x_weight * y_weight;


                resized[i*width*cn + j*cn + k ] = pixel; 
            }

            //printf("Blue value: %d", pixelPtr[i*img.cols*cn +  j*cn + 0] );
        }

    }

}


int main (int argc, char** argv)
{

    std::string image_path = argv[1];
    std::string image_out_path = argv[2]; 
    n_threads = atoi(argv[3]);


    //printf("threads, Time\n");
    struct timeval tval_before, tval_after, tval_result;
    int *retval;

    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    //printf("Mat type: %d", img.type());
    resized = (uint8_t *) malloc(img.channels()*height*width*sizeof(uint8_t));


    gettimeofday(&tval_before, NULL);

    // function call

    //bilinear_resize(img, resized);
    //
    //printf("Create Thread %d\n", i);
    //
    #pragma omp parallel num_threads(n_threads)
    {
        int ID = omp_get_thread_num();

        struct args *Data = (struct args *)malloc(sizeof(struct args));
        Data->img = &img;
        Data->id = ID;
        bilinear_resize(Data);
    }

    // Time calculation 
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);

    printf("%d,%ld.%06ld\n",n_threads,(long int) tval_result.tv_sec, (long int)tval_result.tv_usec);

    // Matrix convertion to Mat 
    Mat resized_img(height, width, CV_8UC(3), resized);
    imshow("Display window", resized_img);
    int k = waitKey(0); // Wait for a keystroke in the window


    if(k == 's')
    {
        imwrite(image_out_path, resized_img);
    }

    return 0;
}

