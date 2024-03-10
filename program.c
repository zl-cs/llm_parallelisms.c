#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double sqr(double x) {
	return x * x;
}

// Generates a sample from the uniform distribution. 
// Assumes that srand() has already been called.
double uniform() {
	return (double) rand() / RAND_MAX;	
}


// Generates a sample from the standard normal distribution.
// Based on the implementation from Numerical Recipies and 
// https://dl.acm.org/doi/pdf/10.1145/138351.138364.
double normal() {
	double u, v, x, y, q;
	do {
		u = uniform();
		v = 1.7156 * (uniform() - 0.5);
		x = u - 0.449871;
		y = fabs(v) + 0.386595;
		q = sqr(x) + y * (0.19600 * y - 0.25472 * x);
	} while (q > 0.27597 
		&& (q > 0.27846 || sqr(v) > -4.0 * log(u) * sqr(u)));
	return v / u;
}


int main(void) {
	srand(time(NULL));

	int n = 100000;

	// Estimate the mean.
	double sum = 0.0;
	for (int i = 0; i < n; i++) {
		sum += normal();
	}		
	double mu = sum / n;

	// Estimate the variance.
	sum = 0.0;
	for (int i = 0; i < n; i++) {
		sum += sqr(normal() - mu);
	}		
	double sig = sum / n;

	printf("mu: %f, sig: %f\n", mu, sig);
	return 0;
}

