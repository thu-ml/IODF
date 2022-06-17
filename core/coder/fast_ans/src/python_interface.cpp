#include "ans.h"

#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
typedef uint64_t symbol_t;

using namespace pybind11::literals;

using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;
typedef Eigen::Array<float, Eigen::Dynamic, 1> ArrayYu;
typedef Eigen::Array<int, Eigen::Dynamic, 1> ArrayYYYu; 


class PyANS {
public:
    PyANS(int ans_mass_bits, int num_streams) {   // num_streams is probably chosen as batchsize
        for (int i = 0; i < num_streams; ++i) {
            m_streams.emplace_back(ans_mass_bits);
        }
    }

    size_t stream_length() {
        size_t total = 0;
        for (const auto& ans : m_streams) {
            total += ans.length();
        }
        return total;
    }

    int num_streams() {
        return m_streams.size();
    }

    void encode_with_pmf_cdf(Eigen::Ref<ArrayYu> pmf, Eigen::Ref<ArrayYu> cdf){
        assert_some_streams();

        int dim = pmf.size();

        parallelize(dim, [&](ANSBitstream& ans, int start, int end) {
            for (int i = end - 1; i >= start; --i) {
                ans.encode(int(pmf(i)), int(cdf(i)));
            }
        });
    }

private:
    std::vector<ANSBitstream> m_streams;

    void assert_some_streams() {
        if (m_streams.empty()) {
            throw std::runtime_error("No ANS streams present");
        }
    }

    // Distribute ANS streams over threads
    void parallelize(int total, const std::function<void(ANSBitstream&, int, int)>& f) {
        #pragma omp parallel for schedule(runtime)
        for (int i_stream = 0; i_stream < m_streams.size(); ++i_stream) {
            // Each stream takes care of a block of data
            int extra = total % m_streams.size();
            int start = (total / m_streams.size()) * i_stream + std::min(i_stream, extra);
            int end = start + (total / m_streams.size()) + (int) (i_stream < extra);
            f(m_streams[i_stream], start, end);
        }
    }

};

PYBIND11_MODULE(fast_ans, m) {
    // extern void test_logistic();
    // m.def("test_logistic", &test_logistic);

    py::class_<PyANS>(m, "ANS")
        .def(py::init<int, int>(), "ans_mass_bits"_a, "num_streams"_a)
        .def("stream_length", &PyANS::stream_length)
        .def("num_streams", &PyANS::num_streams)
        .def("encode_with_pmf_cdf", &PyANS::encode_with_pmf_cdf, "pmf"_a, "cdf"_a)
        // .def("encode_logistic", &PyANS::encode_logistic, "x"_a, "means"_a, "logscales"_a)
        // .def("decode_logistic", &PyANS::decode_logistic, "means"_a, "logscales"_a)
        // .def("encode_mix_logistic", &PyANS::encode_mix_logistic, "x"_a, "means"_a, "logscales"_a, "pi_mix"_a)
        // .def("decode_mix_logistic", &PyANS::decode_mix_logistic, "means"_a, "logscales"_a, "pi_mix"_a)
    ;
}