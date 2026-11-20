
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <string>

using namespace std;

// So luong cum
const int K = 5;

// Do lech chuan cua phan phoi Gaussian
const double STD_DEV = 1.5;

// Toa do tam cua 5 cum
const double centers[K][2] = {
    {-10.0, -10.0},
    {-10.0,  10.0},
    {  0.0,   0.0},
    { 10.0, -10.0},
    { 10.0,  10.0}
};

void generateData(int n, const string& filename) {
    // Khoi tao bo sinh so ngau nhien voi seed co dinh
    mt19937_64 rng(42);

    // Phan phoi chuan N(0, 1.5)
    normal_distribution<double> noise(0.0, STD_DEV);

    ofstream out(filename);

    if (!out.is_open()) {
        cerr << "Cannot create file: " << filename << endl;
        return;
    }

    out << fixed << setprecision(6);

    for (int i = 0; i < n; i++) {
        // Phan bo deu so diem cho 5 cum
        int cluster = i % K;

        double x = centers[cluster][0] + noise(rng);
        double y = centers[cluster][1] + noise(rng);

        out << x << " " << y << "\n";
    }

    out.close();

    cout << "Generated " << n
         << " points -> " << filename << endl;
}

int main() {
    generateData(100000, "data/blobs_100k.txt");

    generateData(500000, "data/blobs_500k.txt");

    generateData(1000000, "data/blobs_1m.txt");

    return 0;
}
