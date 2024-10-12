#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <ranges>
#include <utility>
#include <tuple>
#include <iostream>
#include <stdlib.h>



// recommended changes
// generator should be associated with network

class initialiser{

  protected:
    std::default_random_engine generator;

  public:
    initialiser(){
      generator = std::default_random_engine();
    }
    virtual double generate() = 0;

};

class zeroes : public initialiser{
  public:
    zeroes() : initialiser(){};
    double generate(){
      return 0;
    }
};

class random_uniform: public initialiser{

  public:
    double center, mean;
    std::uniform_real_distribution<double> distribution;

    random_uniform( double center, double mean ) : initialiser(){
      this->center = center;
      this->mean = mean;
      distribution = std::uniform_real_distribution<double>( mean-(center/2.0), mean+(center/2.0));
    }

    double generate(){
      return distribution( generator );
    }

};

class gaussian: public initialiser{

  public:
    double mean, standard_deviation;
    std::normal_distribution<double> distribution;

    gaussian( double mean, double standard_deviation ) : initialiser(){
      this->mean = mean;
      this->standard_deviation = standard_deviation;
      distribution = std::normal_distribution<double>( mean, standard_deviation );
    }

    double generate(){
      return distribution( generator );
    }

};

class activation_function {
  public:
    double (* activation)(double);
    double (* derivative)(double);

    activation_function( double (* activation)(double), double (* derivative)(double) ){
      this->activation = activation;
      this->derivative = derivative;
    }
};

double sigmoid( double x ){ return 1.0/(1.0 + std::exp(-x) ); }
double sigmoid_derivative( double x ){ return sigmoid(x)*( 1 - sigmoid(x) ); }
double relu( double x ){ return std::max( 0.0, x ); }
double relu_derivative( double x ){ return x < 0.0 ? 0.0 : 1.0; }
double leaky_relu( double x ){ return std::max( -0.1*x, x );}
double leaky_relu_derivative( double x ){ return x < 0.0 ? -0.1 : 1.0;  }
double softsign( double x ){ return x/( 1 + std::abs(x) ); }
double softsign_derivative( double x ){ return 1/(  (1 + std::abs(x))*( 1 + std::abs(x)) ); }
double tanh_derivative( double x ){ return (1 - std::tanh(x)*std::tanh(x) ); }

static activation_function sigmoid_activation( sigmoid, sigmoid_derivative );
static activation_function relu_activation( relu, relu_derivative );
static activation_function leaky_relu_activation( leaky_relu, leaky_relu_derivative );
static activation_function softsign_activation( softsign, softsign_derivative );
static activation_function tanh_activation( std::tanh, tanh_derivative );

std::vector<double> softmax( std::vector<double> output ){
  std::ranges::for_each( output, []( double & value ){ value = std::exp(value); } );
  double total = std::accumulate( output.begin(), output.end(), 0.0 );
  std::ranges::for_each( output, [&total]( double & value ){ value /= total; } );
  return output;
}

std::vector<double> square_error( std::vector<double> output, std::vector<double> truth ){
  std::transform( output.cbegin(), output.cend(), truth.cbegin(), output.begin(), []( const double & a, const double & b){ return (a-b)*(a-b); } );
  return output;
}

auto get_column_views( std::vector<double> & matrix, int row_length ){
  return std::ranges::iota_view(0, row_length) | std::views::transform( [&matrix, &row_length ]( int col ){
    return matrix | std::views::drop(col) | std::views::stride(row_length);
  });
}

class layer{
  public:
    int nodes;
    void initialise( layer * previous_layer );
    void feed_forward( std::vector<double> & input, std::vector<double> & output );
    void back_propagate( std::vector<double> & input, std::vector<double> & output,
      std::vector<double> & input_gradient, std::vector<double> & output_gradient,
      const double & learning_rate = 0.15 );
};

class dense_layer: public layer {

  double learning_rate;
  std::vector<double> weights, biases, weight_momentum, bias_momentum;

  public:

    activation_function * activation;
    initialiser * init;

    dense_layer( int nodes, initialiser * init, activation_function * activation, double learning_rate = 0.15 ){
      this->nodes = nodes;
      this->init = init;
      this->activation = activation;
      this->learning_rate = learning_rate;
    }

    void initialise( layer * previous_layer ){

      weights = std::vector<double>( previous_layer->nodes * nodes );
      biases = std::vector<double>( nodes );
      weight_momentum = std::vector<double>( previous_layer->nodes * nodes );
      bias_momentum = std::vector<double>( nodes );

      std::generate_n( weights.begin(), previous_layer->nodes * nodes, 
        [this](){ return init->generate(); } 
      );

      std::generate_n( biases.begin(), nodes, 
        [this](){ return init->generate(); } 
      );

      std::generate_n( weight_momentum.begin(), previous_layer->nodes * nodes, 
        [this](){ return init->generate(); } 
      );

      std::generate_n( bias_momentum.begin(), nodes, 
        [this](){ return init->generate(); } 
      );
    }

    // perform feed forward operation
    std::vector<double> feed_forward( std::vector<double> & input ){

      // initialise output with bias weights
      std::vector<double> output( biases.begin(), biases.end() );

      int input_size = weights.size()/nodes;

      // create an inumerated range of views, each view has a complete set of input weights. 
      auto weight_enum = weights 
        | std::ranges::views::chunk( input_size ) 
        | std::views::enumerate;

      // propagate previous layer forward, index is over output nodes.
      for( auto [ index, weights ] : weight_enum ){
          output[index] += std::inner_product( input.begin(), input.end(), weights.begin(), 0 );
      }

      // apply activation
      std::transform( output.cbegin(), output.cend(), output.begin(), activation->activation );
      
      return output;
    }


    std::vector<double> back_propagate( std::vector<double> input_activation, std::vector<double> output_gradients ){
      
      // create input gradients 
      int input_size = weights.size()/nodes;
      std::vector< double > input_gradients( input_size, 0.0 );
      auto weight_columns = get_column_views( weights, input_size );
      for ( auto[ input_gradient, weight_column ] : std::views::zip( input_gradients, weight_columns ) ){
        input_gradient = std::inner_product( weight_column.cbegin(), weight_column.cend(), output_gradients.cbegin(), 0 );
      }

      // edit weights
      double lr = this->learning_rate;
      auto momentum_columns = get_column_views( weight_momentum, input_size );
      std::vector<double> gradients( nodes );
      for ( auto[ weight_column, momentum_column, input ] : std::views::zip( weight_columns, momentum_columns, input_activation) ){

        // form gradients
        std::transform( output_gradients.cbegin(), output_gradients.cend(), gradients.begin(), 
          [&input]( const double & grad ){ return grad*input; } 
        );
        // modify weight momentums
        std::transform( momentum_column.cbegin(), momentum_column.cend(), gradients.cbegin(), momentum_column.begin(),
          [&lr]( const double & momentum, const double & grad ){ return 0.9*momentum + lr*grad; }
        );
        //  modify weights
        std::transform( weight_column.cbegin(), weight_column.cend(), momentum_column.cbegin(), weight_column.begin(),
          []( const double & weight, const double & momentum ){ return weight - momentum; }
        );

        input = std::inner_product( weight_column.cbegin(), weight_column.cend(), output_gradients.cbegin(), 0 );
      }

      // edit biases
      for ( auto [bias, momentum, output_gradient] : std::views::zip( biases, bias_momentum, output_gradients ) ){
        momentum = 0.9*momentum  + learning_rate * output_gradient;
        bias -= momentum;
      }
      return input_activation;
    }
}; 

class network{

  std::vector<layer *> layers;
  std::vector<double> (*loss)( std::vector<double>, std::vector<double> ); 

  public:

    network(){};
    
    void add_layer( layer * layer ){
      layers.push_back( layer );
    };

    void connect(){
      for (size_t index = 1; index < layers.size(); index++ ) {
        layers[index]->initialise( layers[index-1] );
      }
    };

    void set_loss( std::vector<double> (*loss)( std::vector<double>, std::vector<double> ) = &square_error ){
      this->loss = loss;
    }
    
    void train(){
      // TODO
    }

};

int main(){
  
  zeroes zeroes_initialiser;
  dense_layer dl( 32, &zeroes_initialiser, &sigmoid_activation, 0.10 );
  std::cout << "runs" << std::endl;


};
