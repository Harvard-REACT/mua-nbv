cmake_minimum_required(VERSION 3.8)
project(mua_nbv_planner)

include(CTest)

# ── Build settings ───────────────────────────────────────────────────
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Dependencies ─────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(octomap REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CGAL REQUIRED COMPONENTS Core)
find_package(cv_bridge REQUIRED)

# ── Nodes ────────────────────────────────────────────────────────────

# 1. Ring / Ellipse candidate generator
add_executable(ring_generator_node src/ring_generator_node.cpp)
target_include_directories(ring_generator_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(ring_generator_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(ring_generator_node
  rclcpp geometry_msgs sensor_msgs visualization_msgs
  std_msgs std_srvs builtin_interfaces
  tf2 tf2_ros tf2_eigen cv_bridge
)
target_link_libraries(ring_generator_node Eigen3::Eigen ${OpenCV_LIBRARIES})

# 2. Ellipsoid-projection scorer
add_executable(score_node src/scorer_node.cpp)
target_include_directories(score_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(score_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(score_node
  rclcpp geometry_msgs sensor_msgs std_msgs cv_bridge
  std_srvs tf2 tf2_ros tf2_eigen
)
target_link_libraries(score_node Eigen3::Eigen ${OpenCV_LIBRARIES} CGAL::CGAL OpenMP::OpenMP_CXX)
if(TARGET CGAL::CGAL_Core)
  target_link_libraries(score_node CGAL::CGAL_Core)
endif()

# 3. Voxel map (OctoMap)
add_executable(voxel_map_node src/voxel_map_node.cpp)
target_include_directories(voxel_map_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OCTOMAP_INCLUDE_DIRS}
)
ament_target_dependencies(voxel_map_node
  rclcpp sensor_msgs std_msgs std_srvs
  geometry_msgs visualization_msgs
  tf2 tf2_ros
)
target_link_libraries(voxel_map_node
  Eigen3::Eigen ${OCTOMAP_LIBRARIES} OpenMP::OpenMP_CXX
)

# ── Install ──────────────────────────────────────────────────────────
install(TARGETS ring_generator_node voxel_map_node score_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

# ── Export ───────────────────────────────────────────────────────────
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)

# ── Tests ────────────────────────────────────────────────────────────
if(BUILD_TESTING AND EXISTS "/test/test_candidate_generation.cpp")
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_candidate_generation test/test_candidate_generation.cpp)
  target_include_directories(test_candidate_generation PUBLIC
    1001BUILD_INTERFACE:/include>
    1001INSTALL_INTERFACE:include>
  )
  target_link_libraries(test_candidate_generation Eigen3::Eigen)
endif()

ament_package()
