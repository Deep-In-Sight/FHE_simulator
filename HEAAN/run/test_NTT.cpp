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
	complex<double>* mmult = new complex<double>[n];
	for(long i = 0; i < n; i++) {
		mmult[i] = mvec1[i] * mvec2[i];
	}

	Ciphertext cipher1;
	scheme.encrypt(cipher1, mvec1, n, logp, logq);
	cipher1.np = logn;
	cout << "np: " << logn << endl;
	cout << "np: " << cipher1.np << endl;
	Ciphertext cipher2 = Ciphertext(cipher1);
	//scheme.encrypt(cipher2, mvec1, n, logp, logq);
	//scheme.encrypt(cipher2, mvec2, n, logp, logq);
	
	timeutils.start("MultByVec");
	//scheme.dcrt(cipher1, logp);
	
	// ax -> ra, bx -> rb.  CRT + NTT 

	cout << "CHECK SAME" << endl;
	for (long i = 0; i < 3; i++) {
		cout << i << " " << cipher1.ax[i] << " " << cipher2.ax[i] << " " << endl;
	}
	cout << "            Check  done           \n"  << endl;

	// TARGET
	scheme.multByConstVecAndEqual(cipher2, mvec2, logp);
	cout << "            Cipher2 Done           \n"  << endl;


	
	ring.CRT2(cipher1.ra, cipher1.ax, cipher1.np);
	// After CRT and NTT
	cout << "CRTCRT" << endl;
	for (long i = 0; i < 3; i++) {
		cout << i << " " << cipher1.ra[i] << endl;
	}
	cout << flush() << endl;
	//cout << i << " " << cipher1.ra[i] << " " << cipher2.ra[i] << endl;
	ring.CRT(cipher1.rb, cipher1.bx, cipher1.np);
	cipher1.isCRT = true;



	// uint64_t* rai = cipher1.ra + (i << logN);
	// uint64_t* rbi = cipher1.rb + (i << logN);
	// uint64_t pi = pVec[i];
	// uint64_t pri = prVec[i];
	// _ntl_general_rem_one_struct* red_ss = red_ss_array[i];
	// for (long n = 0; n < N; ++n) {
	// 	rai[n] = _ntl_general_rem_one_struct_apply(a[n].rep, pi, red_ss);
	// 	rbi[n] = _ntl_general_rem_one_struct_apply(b[n].rep, pi, red_ss);
	// }
	// RingMultiplier.NTT(rai, i);
	// ring.NTT(rbi, i);
	cout << " ---- ring.CRT done ----- \n";


	scheme.multByConstVecAndEqual2(cipher1, mvec2, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec1, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec1, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec1, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec1, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec2, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec2, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec2, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec2, logp);
	// scheme.multByConstVecAndEqual2(cipher1, mvec2, logp);
	scheme.INTT(cipher1); 
	cout << "AFTER INTT" << endl;
	for (long i = 0; i < 3; i++) {
		cout << i << " " << cipher1.ra[i] << endl;
	}
	
	scheme.reconstruct(cipher1);
	
	cout << "AFTER ICRT" << endl;
	for (long i = 0; i < 3; i++) {
		cout << i << " " << cipher1.ax[i] << endl;
	}

	cout << "CTXT.ra " << endl;
	for (long i = 0; i < n; i++) {
		cout << i << " " << cipher1.ax[i] << " " << cipher2.ax[i] << endl;
	}

	timeutils.stop("MultByVec");

	complex<double>* dmult1 = scheme.decrypt(secretKey, cipher1);

	cout << "CHECK DMULT1" << endl;
	StringUtils::compare(mmult, dmult1, 3, "mult");

	cout << "CHECK DMULT2" << endl;
	complex<double>* dmult2 = scheme.decrypt(secretKey, cipher1);

	StringUtils::compare(mmult, dmult2, 3, "mult");

	cout << "!!! END TEST MULT By Vec !!!" << endl;
	
}


int main(int argc, char **argv) {
    long logq = 120; ///< Ciphertext Modulus
	long logp = 30; ///< Real message will be quantized by multiplying 2^40
	long logn = 3; ///< log2(The number of slots)
    testMultByVec(logq, logp, logn);
}