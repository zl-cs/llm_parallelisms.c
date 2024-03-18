#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <SDL2/SDL.h>

//Screen dimension constants.
const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;

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


// TODO(eugenhotaj): We can speed this up by using binary search.
double* histogram(double data[], int n_data, int n_bins, double low, double high) {
	double* bins = calloc(n_bins, sizeof(double));
	double step_size = (high - low) / n_bins;
	for (int i = 0; i < n_data; i++) {
		for (int j = 0; j < n_bins; j++) {
			double boundary = low + (j+1) * step_size;
			if (data[i] < boundary || j == n_bins - 1) {
				bins[j] += 1.0 / n_data;
				break;
			}
		}
	}
	return bins;
}

void draw_histogram(double bins[], int n_bins) {
	SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
		"Histogram", 
		SDL_WINDOWPOS_UNDEFINED, 
		SDL_WINDOWPOS_UNDEFINED, 
		SCREEN_WIDTH, 
		SCREEN_HEIGHT, 
		SDL_WINDOW_SHOWN
	);
    SDL_Surface* screenSurface = SDL_GetWindowSurface(window);

	// Draw the histogram.
	SDL_FillRect(screenSurface, NULL, SDL_MapRGB(screenSurface->format, 0xFF, 0xFF, 0xFF));
	double width = (double) SCREEN_WIDTH / n_bins;
	for (int i = 0; i < n_bins; i++) {
		// The height of the histogram bins is drawn proportional to their probability mass. However, 
		// as the number of bins increases, their probability mass will tend to 0 and the overall 
		// histogram size will shrink. To decouple the histogram size from the number of bins we 
		// we rescale the bins so that the middle bin always takes up 75% of the screen.
		double scale = bins[n_bins/2] * 1.25;
		int height = bins[i] * SCREEN_HEIGHT / scale;
		SDL_Rect rect = {
			.x = width * i, 
			.y = SCREEN_HEIGHT - height, 
			.w = i == 0 ? width : round(width * i / i), 
			.h = height
		};
		SDL_FillRect(screenSurface, &rect, SDL_MapRGB(screenSurface->format, 0xFF, 0x00, 0x00));
	}
	SDL_UpdateWindowSurface(window);

	//Hack to get window to stay up.
	SDL_Event e; 
	int quit = 0; 
	while(quit == 0) { 
		while(SDL_PollEvent(&e)){ 
			if(e.type == SDL_QUIT) {
				quit = 1; 
			} 
		}
	}

    SDL_DestroyWindow(window);
    SDL_Quit();
}


int main(void) {
	srand(time(NULL));

	int n = 10000;

	// Estimate the mean and variance.
	double sum = 0.0, ssum = 0.0;
	for (int i = 0; i < n; i++) {
		double sample = normal();
		sum += sample;
		ssum += sqr(sample);
	}	
	double mu = sum / n;
	double sig = (ssum / n) - sqr(mu);
	printf("mu: %f, sig: %f\n", mu, sig);


	// Create a histogram of samples.
	int n_bins = 50;
	double low = -3.0, high = 3.0;
	double samples[n];
	for (int i = 0; i < n; i++) {
		samples[i] = normal();
	}
	double* bins = histogram(samples, n, n_bins, low, high);

	// Draw the histogram.
	draw_histogram(bins, n_bins);

	return 0;
}

