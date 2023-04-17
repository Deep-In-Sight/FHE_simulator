#include "../src/HEAAN.h"

using namespace std;
using namespace NTL;


void testMultByVec(long logq, long logp, long logn) {
	cout << "!!! START TEST MULT By Vec !!!" << endl;

	srand(time(NULL));
	//SetNumThreads(8);
	TimeUtils timeutils;
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);

	long n = (1 << logn);
	complex<double>* mvec1 = EvaluatorUtils::randomComplexArray(n);
	complex<double>* mvec2 = EvaluatorUtils::randomComplexArray(n);
	complex<double>* mvec3 = EvaluatorUtils::randomComplexArray(n);
	complex<double>* mmult = new complex<double>[n];
	
	// Multiply three times
	for(long i = 0; i < n; i++) {
		mmult[i] = mvec1[i] * mvec2[i];
	}

	for(long i = 0; i < n; i++) {
		mmult[i] = mmult[i] * mvec2[i];
	}

	for(long i = 0; i < n; i++) {
		mmult[i] = mmult[i] * mvec3[i];
	}

	Ciphertext cipher1;
	scheme.encrypt(cipher1, mvec1, n, logp, logq);
	Ciphertext cipher2 = Ciphertext(cipher1);
	
	timeutils.start("MultByVec");

	// TARGET
	scheme.multByConstVecAndEqual(cipher2, mvec2, logp);
	scheme.reScaleByAndEqual(cipher2, logp);
	cout << cipher2.logp << "  " << cipher2.logq << endl;
	
	scheme.multByConstVecAndEqual(cipher2, mvec2, logp);
	scheme.reScaleByAndEqual(cipher2, logp);
	cout << cipher2.logp << "  " << cipher2.logq << endl;	

	scheme.multByConstVecAndEqual(cipher2, mvec3, logp);
	timeutils.stop("MultByVec");
	cout << "            Cipher2 Done           \n\n"  << endl;


	timeutils.start("MultByVec 22");
	// ax -> ra, bx -> rb.  CRT + NTT 
	if (cipher1.np == 0){
			cipher1.np = 5;
			cipher1.ra = new uint64_t[cipher1.np << logN];
			cipher1.rb = new uint64_t[cipher1.np << logN];
		}

	ring.CRT2(cipher1.ra, cipher1.ax, cipher1.np);
	ring.CRT2(cipher1.rb, cipher1.bx, cipher1.np);
	cipher1.isCRT = true;

	// Can't use rescale functions 
	// Since valuess are not in .ax, .bx. 
	scheme.multByConstVecAndEqual2(cipher1, mvec2, logp);
	
	cout << cipher1.logp << "  " << cipher1.logq << endl;

	//scheme.reScaleByAndEqual(cipher1, logp);
	//cout << cipher1.logp << "  " << cipher1.logq << endl;
	
	scheme.multByConstVecAndEqual2(cipher1, mvec2, logp);

	cout << cipher1.logp << "  " << cipher1.logq << endl;
	//scheme.reScaleByAndEqual(cipher1, logp);
	
	scheme.multByConstVecAndEqual2(cipher1, mvec3, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec2, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec2, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec2, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec2, logp);
	
	// INTT + ICRT
	scheme.INTT(cipher1);
	scheme.reconstruct(cipher1);
	timeutils.stop("MultByVec 22");
	
	complex<double>* dmult1 = scheme.decrypt(secretKey, cipher1);

	// Non-conversion Ver.
	cout << "CHECK DMULT1" << endl;
	StringUtils::compare(mmult, dmult1, 3, "mult");

	// Original Ver.
	cout << "CHECK DMULT2" << endl;
	complex<double>* dmult2 = scheme.decrypt(secretKey, cipher2);

	StringUtils::compare(mmult, dmult2, 3, "mult");

	cout << "!!! END TEST MULT By Vec !!!" << endl;
	
}


int main(int argc, char **argv) {
    long logq = 150; ///< Ciphertext Modulus
	long logp = 30; ///< Real message will be quantized by multiplying 2^40
	long logn = 15; ///< log2(The number of slots)
    testMultByVec(logq, logp, logn);
}