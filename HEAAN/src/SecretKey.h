/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/
#ifndef HEAAN_SECRETKEY_H_
#define HEAAN_SECRETKEY_H_

#include <NTL/ZZ.h>

#include "Ring.h"

using namespace std;
using namespace NTL;

class SecretKey {
public:

	ZZ* sx = new ZZ[N];

	SecretKey(Ring& ring);
	SecretKey(Ring& ring, std::string path); // save
    SecretKey(std::string path); // load

	//void loadKey(std::string path);

};

#endif
