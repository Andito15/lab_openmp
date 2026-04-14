#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <omp.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <iterator>

using namespace cv;
using namespace std;

// Función para aplicar contraste
Mat applyContrast(const Mat& inputImage, double contrastValue)
{
    Mat outputImage;
    resize(inputImage, outputImage, Size(350, 300));

    #pragma omp parallel for
    for (int y = 0; y < outputImage.rows; ++y)
    {
        for (int x = 0; x < outputImage.cols; ++x)
        {
            for (int c = 0; c < outputImage.channels(); ++c)
            {
                outputImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(
                    contrastValue * (outputImage.at<Vec3b>(y, x)[c] - 128) + 128
                );
            }
        }
    }

    return outputImage;
}

int main()
{
    namespace fs = std::filesystem;
    fs::path imgPath = "imagen.bmp";

    cout << "PWD: " << fs::current_path() << endl;
    cout << "Intentando abrir: " << imgPath << endl;

    if (!fs::exists(imgPath)) {
        cerr << "Archivo NO existe en: " << fs::absolute(imgPath) << endl;
        cerr << "Coloca la imagen en el directorio actual o usa la ruta completa." << endl;
        return -1;
    } else {
        cout << "Archivo existe. Tamano: " << fs::file_size(imgPath) << " bytes" << endl;
    }

    string absPath = fs::absolute(imgPath).string();
    cout << "Ruta absoluta: " << absPath << endl;

    Mat image = imread(absPath, IMREAD_COLOR);

    if (image.empty()) {
        cerr << "imread() returned empty. Intentando imdecode desde buffer..." << endl;

        ifstream fin(absPath, ios::binary);
        if (!fin) {
            cerr << "No se pudo abrir el archivo con ifstream." << endl;
            return -1;
        }

        vector<uchar> buf((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
        cout << "Bytes leidos en buffer: " << buf.size() << endl;

        Mat image2 = imdecode(buf, IMREAD_COLOR);
        if (image2.empty()) {
            cerr << "imdecode() tambien fallo. Probable falta de codecs o archivo corrupto." << endl;
            return -1;
        }

        cout << "imdecode() tuvo exito. Continuando con la imagen decodificada." << endl;
        image = image2;
    }

    double contrast = -1.5;
    Mat result = applyContrast(image, contrast);

    imwrite("resultado.jpg", result);
    cout << "Imagen guardada como 'resultado.jpg'" << endl;

    return 0;
}
