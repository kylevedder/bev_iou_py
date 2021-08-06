#include <math.h>
#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>

#include <boost/geometry.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::python::numpy;
namespace bg = boost::geometry;
using Point = bg::model::point<float, 2, bg::cs::cartesian>;
using Polygon = bg::model::polygon<Point>;

using CornerArray = std::array<Point, 4>;

std::optional<Polygon> BoxToPolygon(const np::ndarray& box_entry) {
  if (box_entry.shape(0) != 5) {
    std::cerr << "Expected boxes of length 5, got length " << box_entry.shape(0)
              << std::endl;
    return {};
  }

  constexpr float kToRad = M_PI / 180.0f;

  const float cx = bp::extract<float>(box_entry[0]);
  const float cy = bp::extract<float>(box_entry[1]);
  const float wx = bp::extract<float>(box_entry[2]);
  const float wy = bp::extract<float>(box_entry[3]);
  const float rot = bp::extract<float>(box_entry[4]);

  // Sanity check inputs for safety.
  if (!(std::isfinite(cx) && std::isfinite(cy) && std::isfinite(wx) &&
        std::isfinite(wy) && std::isfinite(rot))) {
    return {};
  }
  if (wx <= 0 || wy <= 0) {
    return {};
  }

  const float cos_rot = std::cos(rot * kToRad);
  const float sin_rot = std::sin(rot * kToRad);

  const float delta_x = wx * cos_rot + wy * sin_rot;
  const float delta_y = wy * cos_rot + wx * sin_rot;

  return {Polygon(
      {{Point(cx + delta_x, cy + delta_y), Point(cx + delta_x, cy - delta_y),
        Point(cx - delta_x, cy - delta_y), Point(cx - delta_x, cy + delta_y),
        Point(cx + delta_x, cy + delta_y)}})};
}

std::vector<Polygon> BoxesToPolygon(const np::ndarray& boxes) {
  const int count = boxes.shape(0);
  std::vector<Polygon> polys(0);
  polys.reserve(count);
  for (int i = 0; i < count; ++i) {
    const auto opt_poly = BoxToPolygon(bp::extract<np::ndarray>(boxes[i]));
    if (opt_poly) {
      polys.push_back(*opt_poly);
    }
  }
  return polys;
}

float ComputeIntersectionOverUnion(const Polygon& p1, const Polygon& p2) {
  std::vector<Polygon> poly_intersection;
  std::vector<Polygon> poly_union;
  bg::intersection(p1, p2, poly_intersection);
  bg::union_(p1, p2, poly_union);
  const auto inter_area =
      (poly_intersection.empty() ? 0.0f : bg::area(poly_intersection.front()));
  const auto union_area =
      (poly_union.empty() ? 0.0f : bg::area(poly_union.front()));
  return inter_area / union_area;
}

np::ndarray iou(const np::ndarray& boxes, const np::ndarray& qboxes) {
  const int N = boxes.shape(0);
  const int M = qboxes.shape(0);
  auto res_arr =
      np::zeros(bp::make_tuple(N, M), np::dtype::get_builtin<float>());

  const auto boxes_polys = BoxesToPolygon(boxes);
  const auto qboxes_polys = BoxesToPolygon(qboxes);
  for (size_t n = 0; n < boxes_polys.size(); ++n) {
    const auto& box_poly = boxes_polys[n];
    for (size_t m = 0; m < qboxes_polys.size(); ++m) {
      const auto& qbox_poly = qboxes_polys[m];
      res_arr[n][m] = ComputeIntersectionOverUnion(box_poly, qbox_poly);
    }
  }
  return res_arr;
}

BOOST_PYTHON_MODULE(iou_cpp) {
  np::initialize();
  bp::def("iou", iou);
}