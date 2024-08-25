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

double max(double values[], int n_values) {
	double max_ = values[0];
	for (int i = 1; i < n_values; i++) {
		if (values[i] > max_) {
			max_ = values[i];
		}
	}
	return max_;
}

double min(double values[], int n_values) {
	double min_ = values[0];
	for (int i = 1; i < n_values; i++) {
		if (values[i] < min_) {
			min_= values[i];
		}
	}
	return min_;
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
double* histogram(double data[], int n_data, int n_bins) {
	double* bins = calloc(n_bins, sizeof(double));
	double x_min = min(data, n_data);
	double x_max = max(data, n_data);
	double step_size = (x_max - x_min) / n_bins;
	for (int i = 0; i < n_data; i++) {
		for (int j = 0; j < n_bins; j++) {
			double boundary = x_min + (j+1) * step_size;
			if (data[i] < boundary || j == n_bins - 1) {
				bins[j] += 1.0 / n_data;
				break;
			}
		}
	}
	return bins;
}

void draw_histogram(double bins[], int n_bins, double y_min, double y_max) {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
        "Histogram",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        SDL_WINDOW_SHOWN
    );
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    
    // Set background color
    SDL_SetRenderDrawColor(renderer, 240, 240, 240, 255);
    SDL_RenderClear(renderer);

    // Draw axes
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderDrawLine(renderer, 50, SCREEN_HEIGHT - 50, SCREEN_WIDTH - 50, SCREEN_HEIGHT - 50);  // x-axis
    SDL_RenderDrawLine(renderer, 50, 50, 50, SCREEN_HEIGHT - 50);  // y-axis

    double width = (double)(SCREEN_WIDTH - 100) / n_bins;
    int max_height = SCREEN_HEIGHT - 100;

    // Draw histogram bars
    for (int i = 0; i < n_bins; i++) {
        int height = (bins[i] - y_min) / (y_max - y_min) * max_height;
        SDL_Rect rect = {
            .x = 50 + width * i,
            .y = SCREEN_HEIGHT - 50 - height,
            .w = width - 1,  // Leave 1 pixel gap between bars
            .h = height
        };
        SDL_SetRenderDrawColor(renderer, 0, 120, 255, 255);
        SDL_RenderFillRect(renderer, &rect);
        
        // Draw outline
        SDL_SetRenderDrawColor(renderer, 0, 60, 120, 255);
        SDL_RenderDrawRect(renderer, &rect);
    }

    SDL_RenderPresent(renderer);

    // Event loop
    SDL_Event e;
    int quit = 0;
    while (quit == 0) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = 1;
            }
        }
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}


int main(void) {
	srand(time(NULL));

	int n = 10000, n_bins = 50;
	
	// Samples from the standard normal.
	double samples[n];
	for (int i = 0; i < n; i++) {
		samples[i] = normal();
	}
	double* bins = histogram(samples, n, n_bins);
	draw_histogram(bins, n_bins, 0, max(bins, n_bins) * 1.25);


	// Samples from a mixture of two Gaussians.
	double mixture[n];
	for (int i = 0; i < n; i++) {
		if (uniform() < 0.3) {
			mixture[i] = normal() * 1.0 + 2.38;
		} else {
			mixture[i] = normal() * 1.2 - 1.3;
		}
	}
	double* mixture_bins = histogram(mixture, n, n_bins);
	draw_histogram(mixture_bins, n_bins, 0, max(bins, n_bins) * 1.25);

	return 0;
}