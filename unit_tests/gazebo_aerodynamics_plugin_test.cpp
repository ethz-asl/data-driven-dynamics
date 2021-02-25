#include <gtest/gtest.h>

#include <ignition/math.hh>

/* These declarations are needed here because the functions are not
 * public API of the class and therefore not declared in the header
 * of the plugin. */
namespace detail {
ignition::math::Vector3d ThreeAxisRot(
  double r11, double r12, double r21, double r31, double r32);

double NormalizeAbout(double _angle, double reference);

double ShortestAngularDistance(double _from, double _to);

ignition::math::Vector3d QtoZXY(
  const ignition::math::Quaterniond &_q);
}


////////////////////////////////////////////
/// ThreeAxisRot ///////////////////////////
////////////////////////////////////////////

TEST(Sample, Test1) {
    ASSERT_EQ(true, true);
}

