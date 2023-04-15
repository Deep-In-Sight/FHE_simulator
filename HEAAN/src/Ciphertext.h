/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#ifndef HEAAN_CIPHERTEXT_H_
#define HEAAN_CIPHERTEXT_H_

#include <NTL/ZZ.h>

#include <fstream>
#include "Params.h"

using namespace std;
using namespace NTL;


class Ciphertext {
public:

	ZZ* ax = new ZZ[N];
	ZZ* bx = new ZZ[N];

	long logp;
	long logq;
	long n;

	bool isNTT = false;
	bool isCRT = false;

	// np for polymult.
	// NOT for other operations. 
	ZZ* cnstPoly = new ZZ[N];
	long bnd = maxBits(cnstPoly, N);
	long np = ceil((logq + bnd + logN + 2)/(double)pbnd);
	uint64_t* ra = new uint64_t[np << logN];
	uint64_t* rb = new uint64_t[np << logN];


	Ciphertext(long logp = 0, long logq = 0, long n = 0);

	Ciphertext(const Ciphertext& o);

	void copyParams(Ciphertext& o);

	void copy(Ciphertext& o);

	void free();

	long maxBits(const ZZ* f, long n);
	
	virtual ~Ciphertext();
	
};

#endif
