#include "./network.h"

float computeError(Matrix a, Matrix b){
  Matrix tmp = a-b;
  return tmp.norm();

}

void Network::forward(const Matrix& input) {
  if (layers.empty())
    return;
  Matrix host_out, kernel0_out, kernel1_out;
  GpuTimer timer;
  timer.Start();
  layers[0]->cusForward(input, 0);
  timer.Stop();
  float time0 = timer.Elapsed();
  kernel0_out = layers[0]->output();
  
  timer.Start();
  layers[0]->cusForward(input, 1);
  timer.Stop();
  float time1 = timer.Elapsed();
  kernel1_out = layers[0]->output();
  
  
  timer.Start();
  layers[0]->forward(input);
  timer.Stop();
  float timeh = timer.Elapsed();
  host_out = layers[0]->output();
  printf("Convolution 1:\n\n");
  printf("Host\n");
  printf("Host time: %f s\n\n", timeh/1000);

  float err0 = computeError(host_out, kernel0_out);
  printf("Basic GPU Convolution kernel\n");
  printf("Kernel time: %f s\n", time0/1000);
  printf("Error: %f\n\n", err0);
  
  float err1 = computeError(host_out, kernel1_out);
  printf("Using Share Memory GPU Convolution kernel\n");
  printf("Kernel time: %f s\n", time1/1000);
  printf("Error: %f\n", err1);
  printf("-----------------------------------------\n");
  for (int i = 1; i < layers.size(); i++) {
    if(i == 3){
          Matrix host_out, kernel_0_out, kernel_1_out;
          timer.Start();
          layers[i]->cusForward(layers[i-1]->output(), 0);
          timer.Stop();
          float time_0 = timer.Elapsed();
          kernel_0_out = layers[i]->output();
          
          
          timer.Start();
          layers[i]->cusForward(layers[i-1]->output(), 1);
          timer.Stop();
          float time_1 = timer.Elapsed();
          kernel_1_out = layers[i]->output();
          
          timer.Start();
          layers[i]->forward(layers[i-1]->output());
          timer.Stop();
          float timeh = timer.Elapsed();
          host_out = layers[i]->output();
          printf("Convolution 3:\n\n");
          printf("Host\n");
          printf("Host time: %f s\n\n", timeh/1000);
        
          float err_0 = computeError(host_out, kernel_0_out);
          printf("Basic GPU Convolution kernel\n");
          printf("Kernel time: %f s\n", time_0/1000);
          printf("Error: %f\n\n", err_0);
          
          float err_1 = computeError(host_out, kernel_1_out);
          printf("Using Share Memory GPU Convolution kernel\n");
          printf("Kernel time: %f s\n", time_1/1000);
          printf("Error: %f\n", err_1);
          printf("-----------------------------------------\n");
    }
    else {
        layers[i]->forward(layers[i-1]->output());
    }
  }
}

void Network::backward(const Matrix& input, const Matrix& target) {
  int n_layer = layers.size();
  // 0 layer
  if (n_layer <= 0)
    return;
  // 1 layer
  loss->evaluate(layers[n_layer-1]->output(), target);
  if (n_layer == 1) {
    layers[0]->backward(input, loss->back_gradient());
    return;
  }
  // >1 layers
  layers[n_layer-1]->backward(layers[n_layer-2]->output(),
                              loss->back_gradient());
  for (int i = n_layer-2; i > 0; i--) {
    layers[i]->backward(layers[i-1]->output(), layers[i+1]->back_gradient());
  }
  layers[0]->backward(input, layers[1]->back_gradient());
}

void Network::update(Optimizer& opt) {
  for (int i = 0; i < layers.size(); i++) {
    layers[i]->update(opt);
  }
}

std::vector<std::vector<float> > Network::get_parameters() const {
  const int n_layer = layers.size();
  std::vector< std::vector<float> > res;
  res.reserve(n_layer);
  for (int i = 0; i < n_layer; i++) {
    res.push_back(layers[i]->get_parameters());
  }
  return res;
}

void Network::set_parameters(const std::vector< std::vector<float> >& param) {
  const int n_layer = layers.size();
  if (static_cast<int>(param.size()) != n_layer)
      throw std::invalid_argument("Parameter size does not match");
  for (int i = 0; i < n_layer; i++) {
    layers[i]->set_parameters(param[i]);
  }
}

std::vector<std::vector<float> > Network::get_derivatives() const {
  const int n_layer = layers.size();
  std::vector< std::vector<float> > res;
  res.reserve(n_layer);
  for (int i = 0; i < n_layer; i++) {
    res.push_back(layers[i]->get_derivatives());
  }
  return res;
}

void Network::check_gradient(const Matrix& input, const Matrix& target,
                             int n_points, int seed) {
  if (seed > 0)
    std::srand(seed);

  this->forward(input);
  this->backward(input, target);
  std::vector< std::vector<float> > param = this->get_parameters();
  std::vector< std::vector<float> > deriv = this->get_derivatives();

  const float eps = 1e-4;
  const int n_layer = deriv.size();
  for (int i = 0; i < n_points; i++) {
    // Randomly select a layer
    const int layer_id = int(std::rand() / double(RAND_MAX) * n_layer);
    // Randomly pick a parameter, note that some layers may have no parameters
    const int n_param = deriv[layer_id].size();
    if (n_param < 1)  continue;
    const int param_id = int(std::rand() / double(RAND_MAX) * n_param);
    // Turbulate the parameter a little bit
    const float old = param[layer_id][param_id];

    param[layer_id][param_id] -= eps;
    this->set_parameters(param);
    this->forward(input);
    this->backward(input, target);
    const float loss_pre = loss->output();

    param[layer_id][param_id] += eps * 2;
    this->set_parameters(param);
    this->forward(input);
    this->backward(input, target);
    const float loss_post = loss->output();

    const float deriv_est = (loss_post - loss_pre) / eps / 2;

    std::cout << "[layer " << layer_id << ", param " << param_id <<
    "] deriv = " << deriv[layer_id][param_id] << ", est = " << deriv_est <<
    ", diff = " << deriv_est - deriv[layer_id][param_id] << std::endl;

    param[layer_id][param_id] = old;
  }

  // Restore original parameters
  this->set_parameters(param);
}

void Network::save_parameters(std::string filename)
{
    std::vector<std::vector<float>> param = this->get_parameters();
    std::ofstream fout(filename);
    if (!fout.is_open())
        throw std::runtime_error("Cannot open file " + filename);
    for (int i = 0; i < param.size(); i++)
    {
        for (int j = 0; j < param[i].size(); j++)
            fout << param[i][j] << " ";
        fout << std::endl;
    }
    fout.close();
}

void Network::load_parameters(std::string filename)
{
    std::ifstream fin(filename);
    if (!fin.is_open())
        throw std::runtime_error("Cannot open file " + filename);
    std::vector<std::vector<float>> param;
    std::string line;
    while (std::getline(fin, line))
    {
        std::vector<float> row;
        std::stringstream ss(line);
        float val;
        while (ss >> val)
            row.push_back(val);
        param.push_back(row);
    }
    fin.close();
    this->set_parameters(param);
}

