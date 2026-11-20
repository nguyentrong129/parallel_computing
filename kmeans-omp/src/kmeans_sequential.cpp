
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <limits>
#include <chrono>
#include <iomanip>

using namespace std;

// ==================== CONFIGURATION ====================

const int K = 5;                    // Number of clusters
const double EPSILON = 1e-4;        // Convergence threshold
const int MAX_ITER = 100;           // Maximum iterations

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

// Squared Euclidean Distance
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

    // Use the first K points as initial centroids.
    for (int i = 0; i < K; i++) {
        centroids[i] = {points[i].x, points[i].y};
    }

    return centroids;
}

// ==================== SEQUENTIAL K-MEANS ====================

KMeansResult kmeansSequential(const vector<Point>& points) {
    KMeansResult result;

    // Step 1: Initialize centroids
    result.centroids = initializeCentroids(points);

    vector<int> labels(points.size(), -1);

    // Step 2: Iterate until convergence or MAX_ITER
    for (int iter = 0; iter < MAX_ITER; iter++) {

        array<double, K> sumX{};
        array<double, K> sumY{};
        array<long long, K> counts{};

        // Assignment step
        for (size_t i = 0; i < points.size(); i++) {

            int bestCluster = 0;
            double minDistance = numeric_limits<double>::max();

            for (int j = 0; j < K; j++) {
                double distance =
                    squaredDistance(points[i], result.centroids[j]);

                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = j;
                }
            }

            labels[i] = bestCluster;

            // Accumulate coordinates
            sumX[bestCluster] += points[i].x;
            sumY[bestCluster] += points[i].y;
            counts[bestCluster]++;
        }

        // Update step
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

    // Step 3: Reassign points using final centroids
    // and calculate SSE and final cluster sizes

    result.labels.assign(points.size(), -1);
    result.clusterSizes.fill(0);
    result.sse = 0.0;

    for (size_t i = 0; i < points.size(); i++) {

        int bestCluster = 0;
        double minDistance = numeric_limits<double>::max();

        for (int j = 0; j < K; j++) {
            double distance =
                squaredDistance(points[i], result.centroids[j]);

            if (distance < minDistance) {
                minDistance = distance;
                bestCluster = j;
            }
        }

        result.labels[i] = bestCluster;
        result.clusterSizes[bestCluster]++;
        result.sse += minDistance;
    }

    return result;
}

// ==================== RUN EXPERIMENT ====================

void runExperiment(const string& filename) {

    cout << "\n====================================\n";
    cout << "Dataset: " << filename << endl;
    cout << "====================================\n";

    // Read data outside the timed K-Means execution
    vector<Point> points = readData(filename);

    // Measure K-Means execution time
    auto start = chrono::high_resolution_clock::now();

    KMeansResult result = kmeansSequential(points);

    auto end = chrono::high_resolution_clock::now();

    double executionTime =
        chrono::duration<double, milli>(end - start).count();

    // Print results
    cout << fixed << setprecision(6);

    cout << "\nNumber of clusters: " << K << endl;
    cout << "Number of points: " << points.size() << endl;
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
