#include "nbody_header.h"
#include <math.h>
#define PI 3.14159265

#ifdef MPI
//Currently requires nBodies%nRanks == 0
void run_parallel_problem(int nBodies, double dt, int nIters, char * fname)
{
    //Initialize MPI
    int mype, nprocs;
    MPI_File fh;
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    //Number of bodies per rank
    int nlocal = nBodies/nprocs;

    /* Create an MPI type for Body */
	const int num_member = 7;
	MPI_Datatype TYPE_BODY;
	
	MPI_Aint blockoffsets[7];
	blockoffsets[0] = offsetof(Body, x);
	blockoffsets[1] = offsetof(Body, y);
	blockoffsets[2] = offsetof(Body, z);
	blockoffsets[3] = offsetof(Body, vx);
	blockoffsets[4] = offsetof(Body, vy);
	blockoffsets[5] = offsetof(Body, vz);
	blockoffsets[6] = offsetof(Body, mass);
	int blocklength[7] = {1, 1, 1, 1, 1, 1, 1};
	MPI_Datatype types[7] = {MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
	
	MPI_Type_create_struct(num_member, blocklength, blockoffsets, types, &TYPE_BODY);
	MPI_Type_commit(&TYPE_BODY);


//    /* Create an MPI type for position */
//	const int num_member = 3;
//	MPI_Datatype TYPE_POS;
//	
//	MPI_Aint blockoffsets[3];
//	blockoffsets[0] = offsetof(Body, x);
//	blockoffsets[1] = offsetof(Body, y);
//	blockoffsets[2] = offsetof(Body, z);
//	int blocklength[7] = {1, 1, 1};
//	MPI_Datatype types[7] = {MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
//	MPI_Type_create_struct(num_member, blocklength, blockoffsets, types, &TYPE_POS);
//	MPI_Type_commit(&TYPE_POS);



    //Allocate bodies
    Body * locals = (Body *) calloc(nlocal, sizeof(Body));
    Body * incoming = (Body *) calloc(nlocal, sizeof(Body));
    Body * outgoing = (Body *) calloc(nlocal, sizeof(Body));
    Body * swap;

    //Initialize random bodies
    if (mype ==0){
        //For rank 0
        parallel_randomizeBodies(locals, nlocal, mype, nprocs);
        //Send initial sets to all other ranks
        for (int i=1;i<nprocs;i++){
            parallel_randomizeBodies(outgoing, nlocal, mype, nprocs);
            MPI_Send(outgoing, //data
                     nlocal, //count
                     TYPE_BODY, //MPI Datatype
                     i, //Destination rank
                     0, //tag, not used here
                     MPI_COMM_WORLD); //MPI communicator
        }
    }
     else {
         //Receive randomized Bodies from rank 0
         MPI_Recv(locals, //data
                  nlocal, //count
                  TYPE_BODY, //MPI Datatype
                  0, //Source rank (all from 0 for initialization)
                  0, //tag
                  MPI_COMM_WORLD, //MPI communicator
                  MPI_STATUS_IGNORE); //Status
     }
     //Initialize file that all ranks will write to
     initialize_IO(nBodies, nIters, fname, mype);
     //printf("From process %d: Finish initialization\n",mype+1);


     double start = get_time();

     //Loop over all timesteps
     for(int iter = 0; iter < nIters; iter++) {
         distributed_write_timestep(locals, nlocal, iter, nIters, nprocs, mype, &fh);
         //Copy local bodies to incoming for first loop calculation
         memcpy(incoming, locals, nlocal * sizeof(Body));

         //Loop where each iteration will shift the particles over
         //and collective forces are summed
         for (int rank = 0; rank < nprocs; rank++) {
             //Compute new forces and velocites
             compute_forces_multi_set(locals, incoming, dt, nlocal);
             //Change buffers for transfer
             swap = incoming; incoming = outgoing; outgoing = swap;
             //Apply boundary conditions for inormation transfer
             int prev, next;
             if (rank==nprocs-1) next = 0; else next = rank + 1;
             if (rank==0) prev = nprocs - 1; else prev = rank -1;
             MPI_Sendrecv(outgoing, nlocal, TYPE_BODY, next, 0,  //send
                          incoming, nlocal, TYPE_BODY, prev, 0,  //receive
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         }
         //Update positions
         for (int i = 0; i<nlocal; i++) {
             locals[i].x += locals[i].vx*dt;
             locals[i].y += locals[i].vy*dt;
             locals[i].z += locals[i].vz*dt;
                
         }
     }

    // Close data file
	MPI_File_close(&fh);
	MPI_Type_free(&TYPE_BODY);

	double stop = get_time();

	double runtime = stop-start;
	double time_per_iter = runtime / nIters;
	long interactions = nBodies * nBodies;
	double interactions_per_sec = (double) interactions / time_per_iter;

	if (mype == 0) {
		printf("SIMULATION COMPLETE\n");
		printf("Runtime [s]:              %.3le\n", runtime);
		printf("Runtime per Timestep [s]: %.3le\n", time_per_iter);
		printf("interactions:             %ld\n", nIters);
		printf("Interactions per sec:     %.3le\n", interactions_per_sec);
	}

	free(locals);
	free(incoming);
	free(outgoing);
}



void compute_forces_multi_set(Body * local, Body * remote, double dt, int n)
{
	double G = 6.67259e-3;
	double softening = 1.0e-5;

	for (int i = 0; i < n; i++)
	{ 
		double Fx = 0.0;
		double Fy = 0.0;
		double Fz = 0.0;

		// Compute force from all particles in remote
		for (int j = 0; j < n; j++)
		{
			double dx = remote[j].x - local[i].x;
			double dy = remote[j].y - local[i].y;
			double dz = remote[j].z - local[i].z;

			double distance = sqrt(dx*dx + dy*dy + dz*dz + softening);
			double distance_cubed = distance * distance * distance;

			double m_j = remote[j].mass;
			double mGd = G * m_j / distance_cubed;
			Fx += mGd * dx;
			Fy += mGd * dy;
			Fz += mGd * dz;
		}

		local[i].vx += dt*Fx;
		local[i].vy += dt*Fy;
		local[i].vz += dt*Fz;
	}
}

void parallel_randomizeBodies(Body * bodies, int nBodies_per_rank, int mype, int nprocs)
{
	// velocity scaling term
	double vm = 1.0e-7;
    double xdist; double ydist;
    double frequency; double ifrac;
    frequency = (8*PI)/(nBodies_per_rank*nprocs);

    #ifdef OPENMP
    #pragma omp parallel for
    #endif
	for (int i = 0; i < nBodies_per_rank; i++) {
		// Initialize position between -1.0 and 1.0
        // Form spiral thing
        ifrac = (i/(nBodies_per_rank*nprocs));
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
		bodies[i].mass = 0.1 / (nBodies_per_rank*nprocs);
	}

}

// Opens MPI file, and writes header information (nBodies, iterations)
MPI_File initialize_IO(long nBodies, long nIters, char * fname, int mype)
{
	MPI_File fh;
    if (mype==0){
	    FILE * datafile = fopen(fname,"w");
	    fprintf(datafile, "%+.*le %+.*le %+.*le\n", 10, (double)nBodies, 10, (double) nIters, 10, 0.0);
    }
    MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_CREATE|MPI_MODE_RDWR, MPI_INFO_NULL, &fh); 
    printf("From process %d: open file 3\n",mype+1);
	return fh;
}

// Writes all particle locations for a single timestep
void distributed_write_timestep(Body * local_bodies, int nBodies_per_rank, int timestep, int nIters, int nprocs, int mype, MPI_File * fh)
{
    //Calculate offset
    // 54 characters for every particle (3 values, 18 char each)
    int rank_length = nBodies_per_rank*54;
    int timestep_length = rank_length * nprocs;
    MPI_Offset  byte_offset = (timestep_length*timestep+rank_length*mype+54)*sizeof(char);
    char * print_ptr ;
    print_ptr =(char *) malloc(rank_length*sizeof(char));

    for (int i=0; i<nBodies_per_rank;i++){
        sprintf(&print_ptr[54*i],"%+.*le %+.*le %+.*le\n", 10, local_bodies[i].x,
                                                           10, local_bodies[i].y,
                                                           10, local_bodies[i].z);

    }
    MPI_File_set_view(*fh, byte_offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
	MPI_File_write_all(*fh, print_ptr, rank_length, MPI_CHAR, MPI_STATUS_IGNORE);

	free(print_ptr);
}
#endif
