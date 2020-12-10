#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#define MSG_LENGTH 10

using namespace cv;
const int height = 480;
const int width = 854;

int main(int argc, char *argv[])
{
  int tasks, iam;

  std::string image_path = argv[1];
  std::string image_out_path = argv[2];

  Mat img = imread(image_path, IMREAD_COLOR);
  if (img.empty())
  {
    std::cout << "Could not read the image: " << image_path << std::endl;
    return 1;
  }
  uint8_t *pixelPtr = (uint8_t *)img.data;
  int cn = img.channels(), cols = img.cols, rows = img.rows;
  uint8_t *resized = (uint8_t *)malloc(img.channels() * height * width * sizeof(uint8_t));

  MPI_Status status;
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &iam);

  MPI_Bcast(pixelPtr, cn * cols * rows, MPI_UINT8_T, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  //reduction
  int divi = height / tasks;
  int row = (iam * divi);

  uint8_t *_resized = (uint8_t *)malloc(cn * divi * width * sizeof(uint8_t));
  printf("p:%d w:%d h:%d\n", tasks, width, divi);

  float x_ratio = (cols - 1) / (width - 1);
  float y_ratio = (rows - 1) / (height - 1);
  uint8_t a, b, c, d, pixel;
  int idx = 0;
  for(int i = row; i < min(row+divi, height); i++)
  {
    for (int j = 0; j < width; j++)
    {

      int x_l = floor(x_ratio * j), y_l = floor(y_ratio * i);
      int x_h = ceil(x_ratio * j), y_h = ceil(y_ratio * i);

      float x_weight = (x_ratio * j) - x_l;
      float y_weight = (y_ratio * i) - y_l;

      for (int k = 0; k < cn; k++)
      {
        a = pixelPtr[y_l * cols * cn + x_l * cn + k];
        b = pixelPtr[y_l * cols * cn + x_h * cn + k];
        c = pixelPtr[y_h * cols * cn + x_l * cn + k];
        d = pixelPtr[y_h * cols * cn + x_h * cn + k];

        pixel = (a & 0xff) * (1 - x_weight) * (1 - y_weight) + (b & 0xff) * x_weight * (1 - y_weight) + (c & 0xff) * y_weight * (1 - x_weight) + (d & 0xff) * x_weight * y_weight;

        //printf("%d,%d,%d ", i,j,k );
        _resized[(idx) * width * cn + j * cn + k] = pixel;
      }

      //printf("Blue value: %d", pixelPtr[i*img.cols*cn +  j*cn + 0] );
    }
    idx++;
  }

  //   if (iam == 1)
  // {
  //   Mat resized_img(divi, width, CV_8UC(3), _resized);
  //   imshow("Display window", resized_img);
  //   int k = waitKey(0); // Wait for a keystroke in the window

  //   if (k == 's')
  //   {
  //     imwrite(image_out_path, resized_img);
  //   }

  // }

  MPI_Gather(_resized, height * width * cn / tasks, MPI_UINT8_T, resized, height * width * cn / tasks, MPI_UINT8_T, 0, MPI_COMM_WORLD);

  MPI_Finalize();

      if (iam == 0)
  {
    Mat resized_img(height, width, CV_8UC(3), resized);
    imshow("Display window", resized_img);
    int k = waitKey(0); // Wait for a keystroke in the window

    if (k == 's')
    {
      imwrite(image_out_path, resized_img);
    }

  }

}