#include "THSBFloat16.h"

c10::BFloat16 THSBFloat16_ctor(float value)
{
    c10::BFloat16 bf16(value);
    return bf16;
}

float THSBFloat16_op_float(c10::BFloat16 bf16)
{
    return static_cast<float>(bf16);
}

c10::BFloat16 THSBFloat16_op_add(c10::BFloat16 a, c10::BFloat16 b){
    return a + b;
}
c10::BFloat16 THSBFloat16_op_sub(c10::BFloat16 a, c10::BFloat16 b) {
    return a - b;
}
c10::BFloat16 THSBFloat16_op_mul(c10::BFloat16 a, c10::BFloat16 b){
    return a * b;
}
c10::BFloat16 THSBFloat16_op_div(c10::BFloat16 a, c10::BFloat16 b){
    return a / b;
}
float THSBFloat16_op_add_float(c10::BFloat16 a, float b) {
    return a + b;
}
float THSBFloat16_op_sub_float(c10::BFloat16 a, float b) {
    return a - b;
}
float THSBFloat16_op_mul_float(c10::BFloat16 a, float b) {
    return a * b;
}
float THSBFloat16_op_div_float(c10::BFloat16 a, float b) {
    return a / b;
}
float THSBFloat16_op_add_lfloat(float a, c10::BFloat16 b) {
    return a + b;
}
float THSBFloat16_op_sub_lfloat(float a, c10::BFloat16 b) {
    return a - b;
}
float THSBFloat16_op_mul_lfloat(float a, c10::BFloat16 b) {
    return a * b;
}
float THSBFloat16_op_div_lfloat(float a, c10::BFloat16 b) {
    return a / b;
}
double THSBFloat16_op_add_double(c10::BFloat16 a, double b) {
    return a + b;
}
double THSBFloat16_op_sub_double(c10::BFloat16 a, double b) {
    return a - b;
}
double THSBFloat16_op_mul_double(c10::BFloat16 a, double b) {
    return a * b;
}
double THSBFloat16_op_div_double(c10::BFloat16 a, double b) {
    return a / b;
}
double THSBFloat16_op_add_ldouble(double a, c10::BFloat16 b) {
    return a + b;
}
double THSBFloat16_op_sub_ldouble(double a, c10::BFloat16 b) {
    return a - b;
}
double THSBFloat16_op_mul_ldouble(double a, c10::BFloat16 b) {
    return a * b;
}
double THSBFloat16_op_div_ldouble(double a, c10::BFloat16 b) {
    return a / b;
}

c10::BFloat16 THSBFloat16_min(c10::BFloat16 bf16) {
    return std::numeric_limits<c10::BFloat16>::min();
}
c10::BFloat16 THSBFloat16_lowest(c10::BFloat16 bf16){
    return std::numeric_limits<c10::BFloat16>::lowest();
}
c10::BFloat16 THSBFloat16_max(c10::BFloat16 bf16){
    return std::numeric_limits<c10::BFloat16>::max();
}
c10::BFloat16 THSBFloat16_epsilon(c10::BFloat16 bf16){
    return std::numeric_limits<c10::BFloat16>::epsilon();
}
c10::BFloat16 THSBFloat16_round_error(c10::BFloat16 bf16) {
    return std::numeric_limits<c10::BFloat16>::round_error();
}
c10::BFloat16 THSBFloat16_nfinity(c10::BFloat16 bf16) {
    return std::numeric_limits<c10::BFloat16>::infinity();
}
c10::BFloat16 THSBFloat16_quiet_NaN(c10::BFloat16 bf16) {
    return std::numeric_limits<c10::BFloat16>::quiet_NaN();
}
c10::BFloat16 THSBFloat16_signaling_NaN(c10::BFloat16 bf16) {
    return std::numeric_limits<c10::BFloat16>::signaling_NaN();
}
c10::BFloat16 THSBFloat16_denorm_min(c10::BFloat16 bf16) {
    return std::numeric_limits<c10::BFloat16>::denorm_min();
}