add_library(expression_template INTERFACE)
target_include_directories(expression_template INTERFACE
    ${${project_name}_SOURCE_DIR}\\src
    ${${project_name}_SOURCE_DIR}\\src\\expression_template_engine
)
target_compile_options(expression_template INTERFACE -W1 -std:c++17)

