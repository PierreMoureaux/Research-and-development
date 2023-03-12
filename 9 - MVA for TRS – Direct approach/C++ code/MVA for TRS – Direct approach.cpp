#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>

//TRS datas
auto T = 1.0;
auto sTRS = 0.1;

//Model datas
auto Nouter = 100;
auto Ninner = 100;
auto lambB = 0.1;
auto lambC = 0.1;
auto R = 0.4;
auto sIM = 0.02;
auto r = 0.1;
auto sigma = 0.1;
auto deltaT = 0.05;
auto MPOR = 10;
auto S0 = 100;
auto K = 100;
int lT = static_cast<int>(T / deltaT);

template<typename T>
static inline std::vector<T> Quantile(const std::vector<T>& inData, const std::vector<T>& probs)
{
    if (inData.empty())
    {
        return std::vector<T>();
    }

    if (1 == inData.size())
    {
        return std::vector<T>(1, inData[0]);
    }

    std::vector<T> data = inData;
    std::sort(data.begin(), data.end());
    std::vector<T> quantiles;

    for (size_t i = 0; i < probs.size(); ++i)
    {
        T poi = std::lerp<T>(-0.5, data.size() - 0.5, probs[i]);

        size_t left = std::max(int64_t(std::floor(poi)), int64_t(0));
        size_t right = std::min(int64_t(std::ceil(poi)), int64_t(data.size() - 1));

        T datLeft = data.at(left);
        T datRight = data.at(right);

        T quantile = std::lerp<T>(datLeft, datRight, poi - left);

        quantiles.push_back(quantile);
    }

    return quantiles;
}

double MVATRS()
{
    auto timeSum = 0.0;
    //Loop for TRS life cycle
    for (auto k = 0; k != lT; k++)
    {
        auto dfMPOR = exp(-(lT - k - MPOR) * r);
        auto dfk = exp(-(lT - k) * r);
        auto quant = std::vector<double>(Nouter);
        //Outer Monte - Carlo loop
        for (auto j = 0; j != Nouter; j++)
        {
            auto innerSum = std::vector<double>(Ninner);
            //Inner Monte - Carlo loop
            for (auto i = 0; i != Ninner; i++)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::vector<double> phiS;
                for (auto z = 0; z != lT; z++)
                {
                    std::normal_distribution<float> d(0, 1);
                    phiS.emplace_back(d(gen));
                }
                auto S = std::vector<double>(lT);
                S[0] = 100;
                //Specific loop for risk factor generation(here only equity)
                for (auto z = 0; z != (lT - 1); z++)
                {
                    S[z + 1] = S[z] * (r * deltaT + sigma * sqrt(deltaT) * phiS[z] + 1);
                }
                if ((k + MPOR) > lT)
                {
                    innerSum[i] = dfMPOR * (S[lT - 1] - K - sTRS * T) - dfk * (S[k] - K - sTRS * T);
                }
                else
                {
                    innerSum[i] = dfMPOR * (S[k + MPOR - 1] - K - sTRS * T) - dfk * (S[k] - K - sTRS * T);
                }
            }
            quant[j] = Quantile<double>(innerSum,{0.99})[0];
        }
        auto outerSum = std::accumulate(quant.begin(), quant.end(), 0.0) / quant.size();
        timeSum += exp(-k * (lambB + lambC + r)) * outerSum;
    }
    return ((1 - R) * lambB - sIM) * timeSum;
}

int main()
{
    std::cout << "Margin valuation adjustment for bullet fixed rate TRS is " << MVATRS() << std::endl;
}


