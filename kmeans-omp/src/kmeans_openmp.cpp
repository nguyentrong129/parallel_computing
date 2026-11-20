
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iomanip>
#include <omp.h>

using namespace std;

// ==================== CONFIGURATION ====================

const int K = 5;
const double EPSILON = 1e-4;
const int MAX_ITER = 100;

// ==================== DATA STRUCTURES ====================

struct Point {
    double x;
    double y;
};

struct Centroid {
    double x;
    double y;
};

struct KMeansResult {
    array<Centroid, K> centroids;
    vector<int> labels;
    array<long long, K> clusterSizes{};
    double sse = 0.0;
    int iterations = 0;
};

// ==================== READ DATA ====================

vector<Point> readData(const string& filename) {
    ifstream in(filename);

    if (!in.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    vector<Point> points;
    Point p;

    while (in >> p.x >> p.y) {
        points.push_back(p);
    }

    if (points.empty()) {
        throw runtime_error("Dataset is empty: " + filename);
    }

    cout << "Loaded " << points.size()
         << " points from " << filename << endl;

    return points;
}

// ==================== DISTANCE ====================

double squaredDistance(const Point& p, const Centroid& c) {
    double dx = p.x - c.x;
    double dy = p.y - c.y;

    return dx * dx + dy * dy;
}

// ==================== INITIALIZATION ====================

array<Centroid, K> initializeCentroids(
    const vector<Point>& points
) {
    if (points.size() < K) {
        throw runtime_error("Number of points must be >= K");
    }

    array<Centroid, K> centroids;

    // Same initialization as Sequential K-Means
    for (int i = 0; i < K; i++) {
        centroids[i] = {points[i].x, points[i].y};
    }

    return centroids;
}

// ==================== OPENMP K-MEANS ====================

KMeansResult kmeansOpenMP(const vector<Point>& points) {
    KMeansResult result;

    result.centroids = initializeCentroids(points);

    vector<int> labels(points.size(), -1);

    for (int iter = 0; iter < MAX_ITER; iter++) {

        // Global accumulators
        array<double, K> sumX{};
        array<double, K> sumY{};
        array<long long, K> counts{};

        // Parallel region
        #pragma omp parallel
        {
            // Each thread has its own local accumulators
            array<double, K> localSumX{};
            array<double, K> localSumY{};
            array<long long, K> localCounts{};

            // Assignment step
            #pragma omp for schedule(static)
            for (int i = 0; i < static_cast<int>(points.size()); i++) {

                int bestCluster = 0;
                double minDistance =
                    numeric_limits<double>::max();

                for (int j = 0; j < K; j++) {
                    double distance =
                        squaredDistance(points[i], result.centroids[j]);

                    if (distance < minDistance) {
                        minDistance = distance;
                        bestCluster = j;
                    }
                }

                labels[i] = bestCluster;

                // Update thread-local accumulators
                localSumX[bestCluster] += points[i].x;
                localSumY[bestCluster] += points[i].y;
                localCounts[bestCluster]++;
            }

            // Merge local accumulators into global accumulators
            #pragma omp critical
            {
                for (int j = 0; j < K; j++) {
                    sumX[j] += localSumX[j];
                    sumY[j] += localSumY[j];
                    counts[j] += localCounts[j];
                }
            }
        }

        // Update centroids (sequential section)
        double maxMovement = 0.0;

        for (int j = 0; j < K; j++) {

            // Keep old centroid if the cluster is empty
            if (counts[j] == 0) {
                continue;
            }

            Centroid newCentroid = {
                sumX[j] / counts[j],
                sumY[j] / counts[j]
            };

            double dx = newCentroid.x - result.centroids[j].x;
            double dy = newCentroid.y - result.centroids[j].y;

            double movement = sqrt(dx * dx + dy * dy);

            if (movement > maxMovement) {
                maxMovement = movement;
            }

            result.centroids[j] = newCentroid;
        }

        result.iterations = iter + 1;

        // Convergence condition
        if (maxMovement < EPSILON) {
            break;
        }
    }

    // Final assignment and SSE calculation
    result.labels.assign(points.size(), -1);
    result.clusterSizes.fill(0);
    double totalSSE = 0.0;

    #pragma omp parallel for schedule(static) reduction(+:totalSSE)
    for (int i = 0; i < static_cast<int>(points.size()); i++) {

        int bestCluster = 0;
        double minDistance =
            numeric_limits<double>::max();

        for (int j = 0; j < K; j++) {
            double distance =
                squaredDistance(points[i], result.centroids[j]);

            if (distance < minDistance) {
                minDistance = distance;
                bestCluster = j;
            }
        }

        result.labels[i] = bestCluster;

        // Count cluster sizes safely
        #pragma omp atomic
        result.clusterSizes[bestCluster]++;

        totalSSE += minDistance;
    }
    result.sse = totalSSE;

    return result;
}

// ==================== RUN EXPERIMENT ====================

void runExperiment(const string& filename) {

    cout << "\n====================================\n";
    cout << "Dataset: " << filename << endl;
    cout << "====================================\n";

    // Read data outside K-Means execution
    vector<Point> points = readData(filename);

    double start = omp_get_wtime();

    KMeansResult result = kmeansOpenMP(points);

    double end = omp_get_wtime();

    double executionTime = (end - start) * 1000.0;

    cout << fixed << setprecision(6);

    cout << "\nNumber of clusters: " << K << endl;
    cout << "Number of points: " << points.size() << endl;
    cout << "Number of threads: " << omp_get_max_threads() << endl;
    cout << "Iterations: " << result.iterations << endl;
    cout << "SSE: " << result.sse << endl;
    cout << "Execution time: " << executionTime << " ms\n";

    cout << "\nFinal centroids:\n";

    for (int i = 0; i < K; i++) {
        cout << "Cluster " << i
             << ": (" << result.centroids[i].x
             << ", " << result.centroids[i].y << ")"
             << " | Points: " << result.clusterSizes[i]
             << endl;
    }
}

// ==================== MAIN ====================

int main() {
    try {
        runExperiment("data/blobs_100k.txt");
        runExperiment("data/blobs_500k.txt");
        runExperiment("data/blobs_1m.txt");
    }
    catch (const exception& e) {
        cerr << "\nError: " << e.what() << endl;
        return 1;
    }

    return 0;
}
