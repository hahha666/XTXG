#include <torch/torch.h>  // 👈 换成这个全量头文件！
#include <torch/script.h> // 保留用于加载模型
#include <iostream>
#include <vector>

int main() {
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("../resnet18_traced.pt"); 
        std::cout << "模型加载成功！\n";
    } catch (const c10::Error& e) {
        std::cerr << "模型加载失败，请检查路径！\n";
        return -1;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    auto output = module.forward(inputs).toTensor();
    
    std::cout << "前 5 个类别的原始打分: \n";
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    
    return 0;
}