#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include "HEAAN.h"
#include "base64.h"

namespace py = pybind11;

using namespace std;
using namespace NTL;

PYBIND11_MAKE_OPAQUE(complex<double>);
PYBIND11_MAKE_OPAQUE(double);
PYBIND11_MAKE_OPAQUE(NTL::ZZ);

using ComplexDouble = complex<double>;
using Double = double;
using ZZ = NTL::ZZ;

std::string zToString(const ZZ &z)
{
	std::stringstream buffer;
	buffer << z;
	return buffer.str();
}

PYBIND11_MODULE(HEAAN, m)
{
	m.doc() = "HEAAN For Python.";

	// ZZ pointer array
	py::class_<ZZ, std::unique_ptr<ZZ, py::nodelete>>(m, "ZZ")
		.def(py::init<>())
		.def(py::init([](std::uint64_t len) { return new NTL::ZZ[len]; }))
		.def(py::init([](py::array_t<std::int64_t> in) {
			NTL::ZZ *out = new NTL::ZZ[in.size()];
			py::buffer_info in_info = in.request();
			std::int64_t *in_ptr = (std::int64_t *)in_info.ptr;

			NTL_EXEC_RANGE(in_info.size, first, last)
			for (auto i = first; i < last; i++)
			{
				out[i] = NTL::to_ZZ(in_ptr[i]);
			}
			NTL_EXEC_RANGE_END

			return out;
		}))
		.def("__getitem__", [](const ZZ *z, std::int64_t i) { return zToString(z[i]); })
		.def("print", [](const ZZ *vals, long size = 5) {
			std::cout << "[";
			std::cout << vals[0];
			for (long i = 1; i < size; ++i)
			{
				std::cout << ", " << vals[i];
			}
			std::cout << "]" << std::endl;
		});

	// // ComplexDouble
	py::class_<ComplexDouble>(m, "ComplexDouble", py::buffer_protocol())
		.def_buffer([](ComplexDouble *v) -> py::buffer_info {
			return py::buffer_info(
				v,
				sizeof(std::complex<double>),
				py::format_descriptor<std::complex<double>>::format(),
				1,
				{malloc_usable_size(v) / sizeof(std::complex<double>)},
				{sizeof(std::complex<double>)});
		})
		.def(py::init<>())
		.def(py::init([](std::uint64_t len) { return new complex<double>[len]; }))
		.def(py::init([](py::array_t<complex<double>> in) {
			complex<double> *out = new complex<double>[in.size()];
			py::buffer_info in_info = in.request();
			complex<double> *in_ptr = (complex<double> *)in_info.ptr;

			NTL_EXEC_RANGE(in_info.size, first, last)
			for (auto i = first; i < last; i++)
			{
				out[i] = in_ptr[i];
			}
			NTL_EXEC_RANGE_END

			return out;
		}))
        .def("__getarr__",[](const ComplexDouble *z, py::array_t<complex<double>> in) {
			py::buffer_info in_info = in.request();
			complex<double> *in_ptr = (complex<double> *)in_info.ptr;

			NTL_EXEC_RANGE(in_info.size, first, last)
			for (auto i = first; i < last; i++)
			{
				in_ptr[i] = z[i];
			}
			NTL_EXEC_RANGE_END
		})
		.def("__getitem__", [](const ComplexDouble *z, std::int64_t i) { return z[i]; })
		.def("__repr__", [](const ComplexDouble &a) {
			return "(" + to_string(a.real()) + ", " + to_string(a.imag()) + ")";
		})
		.def("print", [](const complex<double> *vals, long size = 5) {
			std::cout << "[";
			std::cout << vals[0];
			for (long i = 1; i < size; ++i)
			{
				std::cout << ", " << vals[i];
			}
			std::cout << "]" << std::endl;
		});

	// Double
	py::class_<Double>(m, "Double", py::buffer_protocol())
		.def_buffer([](Double *v) -> py::buffer_info {
			return py::buffer_info(
				v,
				sizeof(double),
				py::format_descriptor<double>::format(),
				1,
				{malloc_usable_size(v) / sizeof(double)}, // when the length is even, the return from malloc will increase by 1
				{sizeof(double)});
		})
		.def(py::init<>())
		.def(py::init([](std::uint64_t len) { return new double[len]; }))
		.def(py::init([](py::array_t<double> in) {
			double *out = new double[in.size()];
			py::buffer_info in_info = in.request();
			double *in_ptr = (double *)in_info.ptr;

			NTL_EXEC_RANGE(in_info.size, first, last)
			for (auto i = first; i < last; i++)
			{
				out[i] = in_ptr[i];
			}
			NTL_EXEC_RANGE_END

			return out;
		}))
        .def("__getarr__",[](const Double *z, py::array_t<double> in) {
			py::buffer_info in_info = in.request();
			double *in_ptr = (double *)in_info.ptr;

			NTL_EXEC_RANGE(in_info.size, first, last)
			for (auto i = first; i < last; i++)
			{
				in_ptr[i] = z[i];
			}
			NTL_EXEC_RANGE_END
		})
		.def("__getitem__", [](const Double *z, std::int64_t i) { return z[i]; })
		.def("__repr__", [](const Double &a) { return to_string(a); })
		.def("print", [](const double *vals, long size = 5) {
			std::cout << "[";
			std::cout << vals[0];
			for (long i = 1; i < size; ++i)
			{
				std::cout << ", " << vals[i];
			}
			std::cout << "]" << std::endl;
		});

	// TestScheme
	py::class_<TestScheme>(m, "TestScheme")
		// STANDARD TESTS
		.def("testEncrypt", &TestScheme::testEncrypt)
		.def("testEncryptSingle", &TestScheme::testEncryptSingle)
		.def("testAdd", &TestScheme::testAdd)
		.def("testMult", &TestScheme::testMult)
		.def("testiMult", &TestScheme::testiMult)
		// ROTATE & CONJUGATE TESTS
		.def("testRotateFast", &TestScheme::testRotateFast)
		.def("testConjugate", &TestScheme::testConjugate)
		// POWER & PRODUCT TESTS
		.def("testPowerOf2", &TestScheme::testPowerOf2)
		.def("testPower", &TestScheme::testPower)
		// FUNCTION TESTS
		.def("testInverse", &TestScheme::testInverse)
		.def("testLogarithm", &TestScheme::testLogarithm)
		.def("testExponent", &TestScheme::testExponent)
		.def("testExponentLazy", &TestScheme::testExponentLazy)
		.def("testSigmoid", &TestScheme::testSigmoid)
		.def("testSigmoidLazy", &TestScheme::testSigmoidLazy)
		// BOOTSTRAPPING TESTS
		.def("testBootstrap", &TestScheme::testBootstrap)
		.def("testBootstrapSingleReal", &TestScheme::testBootstrapSingleReal)
		.def("testWriteAndRead", &TestScheme::testWriteAndRead);

	// Scheme
	py::class_<Scheme>(m, "Scheme")
		.def(py::init<SecretKey &, Ring &, bool, string>(), py::arg(), py::arg(), py::arg("isSerialized") = false, py::arg("root_path") = "./")
		.def(py::init<Ring &, bool, string>(), py::arg(), py::arg("isSerialized") = true, py::arg("root_path") = "./")
        .def_readonly("_cpp_isSerialized", &Scheme::isSerialized)
        .def_readonly("_cpp_RootPath", &Scheme::RootPath)
        .def_readonly("_cpp_EncKeyName", &Scheme::EncKeyName)
        .def_readonly("_cpp_MulKeyName", &Scheme::MulKeyName)
        .def_readonly("_cpp_ConjKeyName", &Scheme::ConjKeyName)
        .def_readonly("_cpp_RotKeyName", &Scheme::RotKeyName)
		// KEYS GENERATION
		.def("addEncKey", &Scheme::addEncKey)
		.def("addMultKey", &Scheme::addMultKey)
		.def("addConjKey", &Scheme::addConjKey)
		.def("addLeftRotKey", &Scheme::addLeftRotKey)
		.def("addRightRotKey", &Scheme::addRightRotKey)
		.def("addLeftRotKeys", &Scheme::addLeftRotKeys)
		.def("addRightRotKeys", &Scheme::addRightRotKeys)
		.def("addBootKey", &Scheme::addBootKey)
		.def("loadEncKey", &Scheme::loadEncKey)
		.def("loadMultKey", &Scheme::loadMultKey)
		.def("loadConjKey", &Scheme::loadConjKey)
		.def("loadLeftRotKey", &Scheme::loadLeftRotKey)
		.def("loadRightRotKey", &Scheme::loadRightRotKey)
		.def("loadLeftRotKeys", &Scheme::loadLeftRotKeys)
		.def("loadRightRotKeys", &Scheme::loadRightRotKeys)
		.def("loadBootKey", &Scheme::loadBootKey)
		// ENCODING & DECODING
		.def("encode", (void (Scheme::*)(Plaintext &, complex<double> *, long, long, long)) & Scheme::encode)
		.def("encode", (void (Scheme::*)(Plaintext &, double *, long, long, long)) & Scheme::encode)
		.def("decode", (complex<double> * (Scheme::*)(Plaintext &)) & Scheme::decode)
		.def("encodeSingle", (void (Scheme::*)(Plaintext &, complex<double>, long, long)) & Scheme::encodeSingle)
		.def("encodeSingle", (void (Scheme::*)(Plaintext &, double, long, long)) & Scheme::encodeSingle)
		.def("decodeSingle", (complex<double>(Scheme::*)(Plaintext &, double *, long, long, long)) & Scheme::decodeSingle)
		// ENCRYPTION & DECRYPTION
		.def("encryptMsg", (void (Scheme::*)(Ciphertext &, Plaintext &)) & Scheme::encryptMsg)
		.def("decryptMsg", (void (Scheme::*)(Plaintext &, SecretKey &, Ciphertext &)) & Scheme::decryptMsg)
		.def("encrypt", (void (Scheme::*)(Ciphertext &, complex<double> *, long, long, long)) & Scheme::encrypt)
		.def("encrypt", (void (Scheme::*)(Ciphertext &, double *, long, long, long)) & Scheme::encrypt)
		.def("encryptZeros", (void (Scheme::*)(Ciphertext &, long, long, long)) & Scheme::encryptZeros)
		.def("decrypt", (complex<double> * (Scheme::*)(SecretKey &, Ciphertext &)) & Scheme::decrypt)
		.def("encryptSingle", (void (Scheme::*)(Ciphertext &, complex<double>, long, long)) & Scheme::encryptSingle)
		.def("encryptSingle", (void (Scheme::*)(Ciphertext &, double, long, long)) & Scheme::encryptSingle)
		.def("decryptSingle", (complex<double>(Scheme::*)(SecretKey &, Ciphertext &)) & Scheme::decryptSingle)
		// HOMOMORPHIC OPERATIONS
		.def("negate", &Scheme::negate)
		.def("negateAndEqual", &Scheme::negateAndEqual)
		.def("add", &Scheme::add)
		.def("addAndEqual", &Scheme::addAndEqual)
		.def("addConst", (void (Scheme::*)(Ciphertext &, Ciphertext &, double, long)) & Scheme::addConst)
		.def("addConst", (void (Scheme::*)(Ciphertext &, Ciphertext &, NTL::RR &, long)) & Scheme::addConst)
		.def("addConst", (void (Scheme::*)(Ciphertext &, Ciphertext &, complex<double>, long)) & Scheme::addConst)
		.def("addConstAndEqual", (void (Scheme::*)(Ciphertext &, double, long)) & Scheme::addConstAndEqual)
		.def("addConstAndEqual", (void (Scheme::*)(Ciphertext &, NTL::RR &, long)) & Scheme::addConstAndEqual)
		.def("addConstAndEqual", (void (Scheme::*)(Ciphertext &, complex<double>, long)) & Scheme::addConstAndEqual)
		.def("sub", &Scheme::sub)
		.def("subAndEqual", &Scheme::subAndEqual)
		.def("subAndEqual2", &Scheme::subAndEqual2)
		.def("imult", &Scheme::imult)
		.def("idiv", &Scheme::idiv)
		.def("imultAndEqual", &Scheme::imultAndEqual)
		.def("idivAndEqual", &Scheme::idivAndEqual)
		.def("mult", &Scheme::mult)
		.def("multAndEqual", &Scheme::multAndEqual)
		.def("square", &Scheme::square)
		.def("squareAndEqual", &Scheme::squareAndEqual)
		.def("multByConst", (void (Scheme::*)(Ciphertext &, Ciphertext &, double, long)) & Scheme::multByConst)
		.def("multByConst", (void (Scheme::*)(Ciphertext &, Ciphertext &, complex<double>, long)) & Scheme::multByConst)
		.def("multByConstVec", (void (Scheme::*)(Ciphertext &, Ciphertext &, complex<double>*, long)) &Scheme::multByConstVec)
		.def("multByConstVecAndEqual", (void (Scheme::*)(Ciphertext &, complex<double>*, long)) &Scheme::multByConstVecAndEqual)
        .def("multByConstVec", (void (Scheme::*)(Ciphertext &, Ciphertext &, double*, long)) &Scheme::multByConstVec)
		.def("multByConstVecAndEqual", (void (Scheme::*)(Ciphertext &, double*, long)) &Scheme::multByConstVecAndEqual)
		.def("multByConstAndEqual", (void (Scheme::*)(Ciphertext &, double, long)) & Scheme::multByConstAndEqual)
		.def("multByConstAndEqual", (void (Scheme::*)(Ciphertext &, NTL::RR &, long)) & Scheme::multByConstAndEqual)
		.def("multByConstAndEqual", (void (Scheme::*)(Ciphertext &, complex<double>, long)) & Scheme::multByConstAndEqual)
		.def("multByPoly", &Scheme::multByPoly)
		.def("multByPolyNTT", &Scheme::multByPolyNTT)
		.def("multByPolyAndEqual", &Scheme::multByPolyAndEqual)
		.def("multByPolyNTTAndEqual", &Scheme::multByPolyNTTAndEqual)
		.def("multByMonomial", &Scheme::multByMonomial)
		.def("multByMonomialAndEqual", &Scheme::multByMonomialAndEqual)
		.def("leftShift", &Scheme::leftShift)
		.def("leftShiftAndEqual", &Scheme::leftShiftAndEqual)
		.def("doubleAndEqual", &Scheme::doubleAndEqual)
		.def("divByPo2", &Scheme::divByPo2)
		.def("divByPo2AndEqual", &Scheme::divByPo2AndEqual)
		// RESCALING
		.def("reScaleBy", &Scheme::reScaleBy)
		.def("reScaleTo", &Scheme::reScaleTo)
		.def("reScaleByAndEqual", &Scheme::reScaleByAndEqual)
		.def("reScaleToAndEqual", &Scheme::reScaleToAndEqual)
		.def("modDownBy", &Scheme::modDownBy)
		.def("modDownByAndEqual", &Scheme::modDownByAndEqual)
		.def("modDownTo", &Scheme::modDownTo)
		.def("modDownToAndEqual", &Scheme::modDownToAndEqual)
		// ROTATIONS & CONJUGATIONS
		.def("leftRotateFast", &Scheme::leftRotateFast)
		.def("rightRotateFast", &Scheme::rightRotateFast)
		.def("leftRotateFastAndEqual", &Scheme::leftRotateFastAndEqual)
		.def("rightRotateFastAndEqual", &Scheme::rightRotateFastAndEqual)
		.def("conjugate", &Scheme::conjugate)
		.def("conjugateAndEqual", &Scheme::conjugateAndEqual)
		// BOOTSTRAPPING
		.def("normalizeAndEqual", &Scheme::normalizeAndEqual)
		.def("coeffToSlotAndEqual", &Scheme::coeffToSlotAndEqual)
		.def("slotToCoeffAndEqual", &Scheme::slotToCoeffAndEqual)
		.def("exp2piAndEqual", &Scheme::exp2piAndEqual)
		.def("evalExpAndEqual", &Scheme::evalExpAndEqual)
		.def("bootstrapAndEqual", &Scheme::bootstrapAndEqual);

	// BootContext
	py::class_<BootContext>(m, "BootContext")
		.def(py::init<>())
		.def_readwrite("bnd1", &BootContext::bnd1)
		.def_readwrite("bnd2", &BootContext::bnd2)
		.def_readwrite("logp", &BootContext::logp);

	// Ring
	py::class_<Ring>(m, "Ring")
		.def(py::init<>())
		// Encode & Decode
		.def("arrayBitReverse", &Ring::arrayBitReverse)
		.def("EMB", &Ring::EMB)
		.def("EMBInvLazy", &Ring::EMBInvLazy)
		.def("EMBInv", &Ring::EMBInv)
		.def("encode", (void (Ring::*)(NTL::ZZ *, double *, long, long)) & Ring::encode)
		.def("encode", (void (Ring::*)(NTL::ZZ *, complex<double> *, long, long)) & Ring::encode)
		.def("decode", &Ring::decode)
		// CONTEXT
		.def("addBootContext", &Ring::addBootContext)
		// MULTIPLICATION
		.def("maxBits", &Ring::maxBits)
		.def("CRT", &Ring::CRT)
		.def("addNTTAndEqual", &Ring::addNTTAndEqual)
		.def("mult", &Ring::mult)
		.def("multNTT", &Ring::multNTT)
		.def("multDNTT", &Ring::multDNTT)
		.def("multAndEqual", &Ring::multAndEqual)
		.def("multNTTAndEqual", &Ring::multNTTAndEqual)
		.def("square", &Ring::square)
		.def("squareNTT", &Ring::squareNTT)
		.def("squareAndEqual", &Ring::squareAndEqual)
		// OTHER
		.def("mod", &Ring::mod)
		.def("modAndEqual", &Ring::modAndEqual)
		.def("negate", &Ring::negate)
		.def("negateAndEqual", &Ring::negateAndEqual)
		.def("add", &Ring::add)
		.def("addAndEqual", &Ring::addAndEqual)
		.def("sub", &Ring::sub)
		.def("subAndEqual", &Ring::subAndEqual)
		.def("subAndEqual2", &Ring::subAndEqual2)
		.def("multByMonomial", &Ring::multByMonomial)
		.def("multByMonomialAndEqual", &Ring::multByMonomialAndEqual)
		.def("multByConst", &Ring::multByConst)
		.def("multByConstAndEqual", &Ring::multByConstAndEqual)
		.def("leftShift", &Ring::leftShift)
		.def("leftShiftAndEqual", &Ring::leftShiftAndEqual)
		.def("doubleAndEqual", &Ring::doubleAndEqual)
		.def("rightShift", &Ring::rightShift)
		.def("rightShiftAndEqual", &Ring::rightShiftAndEqual)
		// ROTATION & CONJUGATION
		.def("leftRotate", &Ring::leftRotate)
		.def("conjugate", &Ring::conjugate)
		// SAMPLING
		.def("subFromGaussAndEqual",(void (Ring::*)(NTL::ZZ *, const NTL::ZZ&)) &Ring::subFromGaussAndEqual)
        .def("subFromGaussAndEqual",(void (Ring::*)(NTL::ZZ *, const NTL::ZZ&, double)) &Ring::subFromGaussAndEqual)
        .def("addGaussAndEqual",(void (Ring::*)(NTL::ZZ *, const NTL::ZZ&)) &Ring::addGaussAndEqual)
        .def("addGaussAndEqual",(void (Ring::*)(NTL::ZZ *, const NTL::ZZ&, double)) &Ring::addGaussAndEqual)
        //.def("sampleHWT", &Ring::sampleHWT)
        //.def("sampleZO", &Ring::sampleHWT)
		.def("sampleUniform2", &Ring::sampleUniform2);
	// DFT

	// RingMultiplier
	py::class_<RingMultiplier>(m, "RingMultiplier")
		.def(py::init<>())
		.def("primeTest", &RingMultiplier::primeTest)
		.def("NTT", &RingMultiplier::NTT)
		.def("INTT", &RingMultiplier::primeTest)
		.def("CRT", &RingMultiplier::CRT)
		.def("addNTTAndEqual", &RingMultiplier::addNTTAndEqual)
		.def("reconstruct", &RingMultiplier::reconstruct)
		.def("mult", &RingMultiplier::mult)
		.def("multNTT", &RingMultiplier::multNTT)
		.def("multDNTT", &RingMultiplier::multDNTT)
		.def("multAndEqual", &RingMultiplier::multAndEqual)
		.def("multNTTAndEqual", &RingMultiplier::multNTTAndEqual)
		.def("square", &RingMultiplier::square)
		.def("squareNTT", &RingMultiplier::squareNTT)
		.def("squareAndEqual", &RingMultiplier::squareAndEqual)
		.def("mulMod", &RingMultiplier::mulMod)
		.def("mulModBarrett", &RingMultiplier::mulModBarrett)
		.def("butt", &RingMultiplier::butt)
		.def("ibutt", &RingMultiplier::ibutt)
		.def("idivN", &RingMultiplier::idivN)
		.def("invMod", &RingMultiplier::invMod)
		.def("powMod", &RingMultiplier::powMod)
		.def("inv", &RingMultiplier::inv)
		.def("pow", &RingMultiplier::pow)
		.def("bitReverse", &RingMultiplier::bitReverse)
		.def("findPrimeFactors", &RingMultiplier::findPrimeFactors)
		.def("findPrimitiveRoot", &RingMultiplier::findPrimitiveRoot)
		.def("findMthRootOfUnity", &RingMultiplier::findMthRootOfUnity);

	// Ciphertext
	py::class_<Ciphertext>(m, "Ciphertext")
		.def(py::init<long, long, long>(), py::arg("logp") = 0, py::arg("logq") = 0, py::arg("n") = 0)
		.def(py::init<const Ciphertext &>())
		.def("copyParams", &Ciphertext::copyParams)
		.def("copy", &Ciphertext::copy)
		.def("free", &Ciphertext::free)
		.def_readwrite("logp", &Ciphertext::logp)
		.def_readwrite("logq", &Ciphertext::logq)
		.def_readwrite("n", &Ciphertext::n)
		.def("__repr__", [](const Ciphertext &p) {
			return "<class.Ciphertext logp: "+to_string(p.logp)+" logq: "+to_string(p.logq)+" n: "+to_string(p.n)+">";
		})
		.def(py::pickle(
			[](const Ciphertext &c) { // __getstate_
				std::stringstream output(std::ios::binary | std::ios::out);
				long np = ceil(((double)c.logq + 1)/8);
				ZZ q = conv<ZZ>(1) << c.logq;
				unsigned char* bytes = new unsigned char[np];
				for (long i = 0; i < N; ++i) {
					c.ax[i] %= q;
					BytesFromZZ(bytes, c.ax[i], np);
					output.write(reinterpret_cast<char*>(bytes), np);
				}
				for (long i = 0; i < N; ++i) {
					c.bx[i] %= q;
					BytesFromZZ(bytes, c.bx[i], np);
					output.write(reinterpret_cast<char*>(bytes), np);
				}
				std::string cipherstr = output.str();
				std::string encoded_cipher = base64_encode(reinterpret_cast<const unsigned char *>(cipherstr.c_str()), (unsigned int)cipherstr.length());

				return py::make_tuple(c.logp, c.logq, c.n, encoded_cipher);
			},
			[](py::tuple t) { // __setstate__
				if (t.size() != 4)
					throw std::runtime_error("Invalid state!");

				long logp = t[0].cast<int>();
				long logq = t[1].cast<int>();
				long n = t[2].cast<int>();
				Ciphertext cipher(logp, logq, n);

				std::string encoded_cipher = t[3].cast<string>();
				std::string cipherstr_decoded = base64_decode(encoded_cipher);
				std::stringstream input(std::ios::binary | std::ios::in);
				input.str(cipherstr_decoded);
				long np = ceil(((double)logq + 1)/8);
				unsigned char* bytes = new unsigned char[np];
				for (long i = 0; i < N; ++i) {
					input.read(reinterpret_cast<char*>(bytes), np);
					ZZFromBytes(cipher.ax[i], bytes, np);
				}
				for (long i = 0; i < N; ++i) {
					input.read(reinterpret_cast<char*>(bytes), np);
					ZZFromBytes(cipher.bx[i], bytes, np);
				}

				return cipher;
			}));

    // Plaintext. No serialization implemented yet.
    py::class_<Plaintext>(m, "Plaintext")
		.def(py::init<long, long, long>(), py::arg("logp") = 0, py::arg("logq") = 0, py::arg("n") = 0)
		.def_readwrite("logp", &Plaintext::logp)
		.def_readwrite("logq", &Plaintext::logq)
		.def_readwrite("n", &Plaintext::n)
		.def("__repr__", [](const Plaintext &p) {
			return "<class.Plaintext logp: "+to_string(p.logp)+" logq: "+to_string(p.logq)+" n: "+to_string(p.n)+">";
		});

	// EvaluatorUtils
	py::class_<EvaluatorUtils>(m, "EvaluatorUtils")
		// RANDOM REAL AND COMPLEX NUMBERS
		.def_static("randomReal", &EvaluatorUtils::randomReal, py::arg("bound") = 1.0)
		.def_static("randomComplex", &EvaluatorUtils::randomComplex, py::arg("bound") = 1.0)
		.def_static("randomCircle", &EvaluatorUtils::randomCircle, py::arg("anglebound") = 1.0)
		.def_static("randomRealArray", &EvaluatorUtils::randomRealArray, py::arg(), py::arg("bound") = 1.0)
		.def_static("randomComplexArray", &EvaluatorUtils::randomComplexArray, py::arg(), py::arg("bound") = 1.0)
		// .def("randomComplexArray", [](long size, double bound = 1.0){ return EvaluatorUtils::randomComplexArray(size, bound);})
		.def_static("randomCircleArray", &EvaluatorUtils::randomCircleArray, py::arg(), py::arg("bound") = 1.0)
		// DOUBLE & RR <-> ZZ
		.def_static("scaleDownToReal", &EvaluatorUtils::scaleDownToReal)
		// .def_static("scaleUpToZZ", (static ZZ (EvaluatorUtils::*)(const double, const long)) &EvaluatorUtils::scaleUpToZZ);
		.def("scaleUpToZZ", [](const double x, const long logp) { return EvaluatorUtils::scaleUpToZZ(x, logp); })
		.def("scaleUpToZZ", [](const NTL::RR &x, const long logp) { return EvaluatorUtils::scaleUpToZZ(x, logp); })
		// ROTATIONS
		.def_static("leftRotateAndEqual", &EvaluatorUtils::leftRotateAndEqual)
		.def_static("rightRotateAndEqual", &EvaluatorUtils::rightRotateAndEqual);

	// SchemeAlgo
	py::class_<SchemeAlgo>(m, "SchemeAlgo")
		.def(py::init<Scheme &>())
		.def("powerOf2", &SchemeAlgo::powerOf2)
		.def("powerOf2Extended", &SchemeAlgo::powerOf2Extended)
		.def("power", &SchemeAlgo::power)
		.def("powerExtended", &SchemeAlgo::powerExtended)
		.def("inverse", &SchemeAlgo::inverse)
		.def("function", &SchemeAlgo::function)
		.def("functionLazy", &SchemeAlgo::functionLazy)
        .def("function_poly", &SchemeAlgo::function_poly)
		.def("functionLazy_poly", &SchemeAlgo::functionLazy_poly);

	// SecretKey
	py::class_<SecretKey>(m, "SecretKey")
		.def(py::init<Ring &>())
        .def(py::init<Ring &, string>(), py::arg(), py::arg("path") = "sk.dat")
        .def(py::init<string>(), py::arg("path") = "sk.dat");

	// StringUtils
	py::class_<StringUtils>(m, "StringUtils")
		// SHOW ARRAY
		.def_static("showVec", (void (*)(long *, long)) & StringUtils::showVec)
		.def_static("showVec", (void (*)(double *, long)) & StringUtils::showVec)
		.def_static("showVec", (void (*)(complex<double> *, long)) & StringUtils::showVec)
		.def_static("showVec", (void (*)(NTL::ZZ *, long)) & StringUtils::showVec)
		// SHOW & COMPARE ARRAY
		.def_static("compare", (void (*)(double, double, string)) & StringUtils::compare)
		.def_static("compare", (void (*)(complex<double>, complex<double>, string)) & StringUtils::compare)
		.def_static("compare", (void (*)(double *, double *, long, string)) & StringUtils::compare)
		.def_static("compare", (void (*)(complex<double> *, complex<double> *, long, string)) & StringUtils::compare)
		.def_static("compare", (void (*)(double *, double, long, string)) & StringUtils::compare)
		.def_static("compare", (void (*)(complex<double> *, complex<double>, long, string)) & StringUtils::compare)
		.def_static("compare", (void (*)(double, double *, long, string)) & StringUtils::compare)
		.def_static("compare", (void (*)(complex<double>, complex<double> *, long, string)) & StringUtils::compare);

	// TimeUtils
	py::class_<TimeUtils>(m, "TimeUtils")
		.def(py::init<>())
		.def("start", &TimeUtils::start)
		.def("stop", &TimeUtils::stop);

	py::class_<SerializationUtils>(m, "SerializationUtils")
		.def_static("writeCiphertext", &SerializationUtils::writeCiphertext)
		.def_static("readCiphertext", &SerializationUtils::readCiphertext)
		.def_static("writeKey", &SerializationUtils::writeKey)
		.def_static("readKey", &SerializationUtils::readKey);
}
