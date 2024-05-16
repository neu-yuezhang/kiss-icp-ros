// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "Registration.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <cmath>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

namespace {

inline double square(double x) { return x * x; }

struct ResultTuple {
    ResultTuple() {
        JTJ.setZero();
        JTr.setZero();
    }

    ResultTuple operator+(const ResultTuple &other) {
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        return *this;
    }

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
};

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}

constexpr int MAX_NUM_ITERATIONS_ = 500;
constexpr double ESTIMATION_THRESHOLD_ = 0.0001;

std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem(
    const std::vector<Eigen::Vector3d> &source,
    const std::vector<Eigen::Vector3d> &target,
    double kernel) {
    auto compute_jacobian_and_residual = [&](auto i) {
        const Eigen::Vector3d residual = source[i] - target[i];
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source[i]);
        return std::make_tuple(J_r, residual);
    };

    const auto &[JTJ, JTr] =
        tbb::parallel_reduce(  // 使用了TBB库中的parallel_reduce函数来实现并行计算和归约操作
                               // Range
            tbb::blocked_range<size_t>{
                0, source.size()},  // 这个范围将被分割成多个子范围，每个子范围将由一个线程并行处理
            // Identity

            // 第一个Lambda表达式：这个Lambda表达式定义了并行计算的逻辑。
            // 它接受一个tbb::blocked_range参数和一个ResultTuple参数，
            // 并返回一个ResultTuple结果。在每个并行计算的迭代中，它会计算Jacobian矩阵和残差，
            // 并根据权重进行累加。这个Lambda表达式使用了auto关键字来推导变量的类型，并使用了引用捕获（&）来获取外部变量的引用。

            // 第二个Lambda表达式：这个Lambda表达式定义了并行归约的逻辑。
            // 它接受两个ResultTuple参数，并返回一个ResultTuple结果。
            // 在每个并行归约的迭代中，它会将两个ResultTuple相加，将结果返回。这个Lambda表达式使用了引用捕获（&）来获取外部变量的引用。
            ResultTuple(),
            // 1st Lambda: Parallel computation
            [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
                auto Weight = [&](double residual2) {
                    return square(kernel) / square(kernel + residual2);
                };
                auto &[JTJ_private, JTr_private] = J;
                for (auto i = r.begin(); i < r.end(); ++i) {
                    const auto &[J_r, residual] = compute_jacobian_and_residual(i);
                    const double w = Weight(residual.squaredNorm());
                    JTJ_private.noalias() += J_r.transpose() * w * J_r;
                    JTr_private.noalias() += J_r.transpose() * w * residual;
                }
                return J;
            },
            // 2nd Lambda: Parallel reduction of the private Jacboians
            [&](ResultTuple a, const ResultTuple &b) -> ResultTuple { return a + b; });

    return std::make_tuple(JTJ, JTr);
}
}  // namespace

namespace kiss_icp {

Sophus::SE3d RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                           const VoxelHashMap &voxel_map,
                           const Sophus::SE3d &initial_guess,
                           double max_correspondence_distance,
                           double kernel) {
    if (voxel_map.Empty()) return initial_guess;

    // Equation (9)
    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    // ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < MAX_NUM_ITERATIONS_; ++j) {
        // Equation (10)
        const auto &[src, tgt] = voxel_map.GetCorrespondences(source, max_correspondence_distance);
        // Equation (11)
        const auto &[JTJ, JTr] = BuildLinearSystem(src, tgt, kernel);
        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        // Equation (12)
        TransformPoints(estimation, source);
        // Update iterations
        T_icp = estimation * T_icp;
        // Termination criteria
        if (dx.norm() < ESTIMATION_THRESHOLD_) break;
    }
    // Spit the final transformation
    return T_icp * initial_guess;
}

}  // namespace kiss_icp
