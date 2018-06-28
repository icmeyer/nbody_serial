#include "nbody_header.h"
#include <math.h>
#define PI 3.14159265

void run_serial_problem(int nBodies, double dt, int nIters, char * fname)
{
	// Open File and Write Header Info
	FILE * datafile = fopen("nbody.dat","w");
	fprintf(datafile, "%+.*le %+.*le %+.*le\n", 10, (double)nBodies, 10, (double) nIters, 10, 0.0);

	// Allocate Bodies
	Body * bodies  = (Body *) calloc( nBodies, sizeof(Body) );
	randomizeBodies(bodies, nBodies);

	double start = get_time();

	// Loop over timesteps
	for (int iter = 0; iter < nIters; iter++)
	{
		printf("iteration: %d\n", iter);

		// Output body positions to file
		for (int i = 0; i < nBodies; i++)
			fprintf(datafile, "%+.*le %+.*le %+.*le\n", 10, bodies[i].x, 10, bodies[i].y, 10, bodies[i].z);

		// Compute new forces & velocities for all particles
		compute_forces(bodies, dt, nBodies);

		// Update positions of all particles
		for (int i = 0 ; i < nBodies; i++)
		{
			bodies[i].x += bodies[i].vx*dt;
			bodies[i].y += bodies[i].vy*dt;
			bodies[i].z += bodies[i].vz*dt;
		}

	}

	// Close data file
	fclose(datafile);

	double stop = get_time();

	double runtime = stop-start;
	double time_per_iter = runtime / nIters;
	long interactions = nBodies * nBodies;
	double interactions_per_sec = interactions / time_per_iter;

	printf("SIMULATION COMPLETE\n");
	printf("Runtime [s]:              %.3le\n", runtime);
	printf("Runtime per Timestep [s]: %.3le\n", time_per_iter);
	printf("Interactions per sec:     %.3le\n", interactions_per_sec);

	free(bodies);
}

// Randomizes all bodies to the following default criteria
// Locations (uniform random between -1.0 < r < 1.0 )
// Velocities (uniform random between -1.0e-3 < r < 1.0e3 )
// Now it makes a spiral thing!
// Masses (all equal at 1.0 / nBodies)
// You should make this more exotic
void randomizeBodies(Body * bodies, int nBodies)
{
	// velocity scaling term
	double vm = 1.0e-7;
    double xdist; double ydist;
    double frequency; double ifrac;
    frequency = (8*PI)/nBodies;

	for (int i = 0; i < nBodies; i++) {
		// Initialize position between -1.0 and 1.0
        // Form spiral thing
        ifrac = (i/nBodies);
        xdist = 1.0; ydist = 1.0;
		bodies[i].x = xdist*sin(ifrac*frequency);
		bodies[i].y = ydist*sin(ifrac*frequency);
		bodies[i].z = 2.0 * ifrac - 0.99;

		// Intialize velocities
		bodies[i].vx = (2.0*vm * (rand() / (double)RAND_MAX) - vm);
		bodies[i].vy = (2.0*vm * (rand() / (double)RAND_MAX) - vm);
		bodies[i].vz = (2.0*vm * (rand() / (double)RAND_MAX) - vm);

		// Initialize masses so that total mass of system is constant
		// regardless of how many bodies are simulated
		bodies[i].mass = 0.1 / nBodies;
	}
}

// Computes the forces between all bodies and updates
// their velocities accordingly
void compute_forces(Body * bodies, double dt, int nBodies)
{
	double G = 6.67259e-3;
	double softening = 1.0e-5;

	// For each particle in the set
	for (int i = 0; i < nBodies; i++)
	{ 
		double Fx = 0.0;
		double Fy = 0.0;
		double Fz = 0.0;

		// Compute force from all other particles in the set
		for (int j = 0; j < nBodies; j++)
		{
			// F_ij = G * [ (m_i * m_j) / distance^3 ] * (location_j - location_i) 

			// First, compute the "location_j - location_i" values for each dimension
			double dx = bodies[j].x - bodies[i].x;
			double dy = bodies[j].y - bodies[i].y;
			double dz = bodies[j].z - bodies[i].z;

			// Then, compute the distance^3 value
			// We will also include a "softening" term to prevent near infinite forces
			// for particles that come very close to each other (helps with stability)

			// distance = sqrt( dx^2 + dx^2 + dz^2 )
			double distance = sqrt(dx*dx + dy*dy + dz*dz + softening);
			double distance_cubed = distance * distance * distance;

			// Now compute G * m_2 * 1/distance^3 term, as we will be using this
			// term once for each dimension
			// NOTE: we do not include m_1 here, as when we compute the change in velocity
			// of particle 1 later, we would be dividing this out again, so just leave it out
			double m_j = bodies[j].mass;
			double mGd = G * m_j / distance_cubed;
			Fx += mGd * dx;
			Fy += mGd * dy;
			Fz += mGd * dz;
		}

		// With the total forces on particle "i" known from this batch, we can then update its velocity
		// v = (F * t) / m_i
		// NOTE: as discussed above, we have left out m_1 from previous velocity computation,
		// so this is left out here as well
		bodies[i].vx += dt*Fx;
		bodies[i].vy += dt*Fy;
		bodies[i].vz += dt*Fz;
	}
}
