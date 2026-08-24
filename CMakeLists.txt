cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerToothOrthogonalSlices)

# Extension metadata
set(EXTENSION_HOMEPAGE "https://github.com/agporto/SlicerToothOrthogonalSlices")
set(EXTENSION_CATEGORY "SlicerMorph")
set(EXTENSION_CONTRIBUTORS "Arthur Porto (Florida Museum of Natural History, University of Florida)")
set(EXTENSION_DESCRIPTION "Estimate an editable curved geometric axis through a tooth CT and export local orthogonal cross-sections with spatial metadata and quality control.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/agporto/SlicerToothOrthogonalSlices/main/ToothOrthogonalSlices/Resources/Icons/ToothOrthogonalSlices.png")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA")
set(EXTENSION_BUILD_SUBDIRECTORY ".")
set(EXTENSION_STATUS "Experimental")
set(EXTENSION_LICENSE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)
set(EXTENSION_README_FILE ${CMAKE_CURRENT_SOURCE_DIR}/README.md)

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(ToothOrthogonalSlices)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
