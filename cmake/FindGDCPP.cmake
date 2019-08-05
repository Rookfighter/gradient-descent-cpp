# FindGDCPP.txt
#
#     Author: Fabian Meyer
# Created On: 05 Aug 2019
#
# Defines
#   GDCPP_INCLUDE_DIR
#   GDCPP_FOUND

find_path(GDCPP_INCLUDE_DIR
    HINTS
    ${GDCPP_ROOT}
    ENV GDCPP_ROOT
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GDCPP DEFAULT_MSG GDCPP_INCLUDE_DIR)
