#include <assert.h>
#include <math.h>
#include "uwnet.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i*m.cols + j];
            if(a == LOGISTIC){
                m.data[i * m.cols + j] = 1.0 / (1.0 + exp(-x));
            } else if (a == RELU){
                if (x <= 0) {
                    m.data[i * m.cols + j] = 0;
                }
            } else if (a == LRELU){
                if (x <= 0) {
                    m.data[i * m.cols + j] = 0.1 * x;
                }
            } else if (a == SOFTMAX){
                m.data[i * m.cols + j] = exp(x);
            }
            sum += m.data[i*m.cols + j];
        }
        if (a == SOFTMAX) {
            for (int col = 0; col < m.cols; col++) {
                double x = m.data[i * m.cols + col];
                m.data[i * m.cols + col] = x / sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    float *ddata = d.data;
    for (i = 0; i < m.rows; ++i){
        for (j = 0; j < m.cols; ++j){
            double x = m.data[i*m.cols + j];
            // potentially much faster to put ifs somewhere else
            if (a == LOGISTIC) {
                *ddata = (*ddata) * x * (1 - x);
            } else if (a == RELU) {
                if (x <= 0) {
                    *ddata = 0.0;
                }
            } else if (a == LRELU) {
                if (x <= 0) {
                    *ddata = 0.1 * (*ddata);
                }
            }
            ddata++;
        }
    }
}
