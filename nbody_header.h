#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#ifdef MPI
#include <mpi.h>
#endif
#ifdef OPENMP
#include <omp.h>
#endif

typedef struct {
	// Location r_i = (x,y,z)
	double x;
	double y;
	double z;
	// Velocity v_i = (vx, vy, vz)
	double vx;
	double vy;
	double vz;
	// Mass
	double mass;
} Body;

//Used in writing output, only need position
typedef struct {
	// Location r_i = (x,y,z)
	double x;
	double y;
	double z;
} BodyPosOnly;

// serial.c
void run_serial_problem(int nBodies, double dt, int nIters, char * fname);
void randomizeBodies(Body * bodies, int nBodies);
void compute_forces(Body * bodies, double dt, int nBodies);

// utils.c
double get_time(void);
void print_inputs(long nBodies, double dt, int nIters, int nthreads );

// parallel.c
#ifdef MPI
void run_parallel_problem(int nBodies, double dt, int nIters, char * fname);
void compute_forces_multi_set(Body * local, Body * remote, double dt, int n);
void parallel_randomizeBodies(Body * bodies, int nBodies_per_rank, int mype, int nprocs);
MPI_File initialize_IO(long nBodies, long nIters, char * fname, int mype);
//void distributed_write_timestep(Body * local_bodies, long nBodies_per_rank, int timestep, int nIters, int nprocs, int mype, MPI_File * fh);
void distributed_write_timestep(Body * local_bodies, int nBodies_per_rank, int timestep, int nIters, int nprocs, int mype, MPI_File * fh);
#endif
