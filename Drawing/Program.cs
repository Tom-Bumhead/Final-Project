using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Windows.Input;

/* TO DO:
 *  COMMENT YOUR DAMN CODE
 */

namespace NeuralNetworkLearning
{
    class Program
    {

        static (Matrix<double>, Matrix<double>, Vector<double>) DigitImageToInput(List<DigitImage> input)
        {
            shuffle(input); //shuffles inputs to make sure no patterns arise
            Matrix<double> x = CreateMatrix.Dense<double>(input.Count, 784);    //creates input and true
            Vector<double> true_s = CreateVector.Dense<double>(input.Count);
            int counter = 0;
            foreach (var picture in input)
            {
                Vector<double> temp = CreateVector.Dense<double>(784);
                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                    {
                        temp[(i * 28) + j] = picture.pixels[i][j] / 255.0;  //normalise greyscale value and set each input to pixel value for that vector of the batch
                    }
                }
                x.SetRow(counter, temp);

                true_s[counter] = picture.label; //sets sparse label for correct output
                counter++;
            }

            double[,] trueOH = new double[input.Count, 10];
            for (int i = 0; i < true_s.Count; i++)
            {
                trueOH[i, (int)true_s[i]] = 1;  //set One Hot label for correct output
            }
            Matrix<double> true_oh = CreateMatrix.DenseOfArray(trueOH);
            return (x, true_oh, true_s); //returns inputs and true values
        }

        static int displayMenu()
        {
            Console.Clear();    //displays menu for user choice
            Console.Write("======= Welcome Tom's Neural Network =======\n" +
                              "     1. New Network\n" +
                              "     2. Load Network\n" +
                              ">>> ");
            return GetInt(1, 2);
        }

        public static int GetInt(int min = int.MinValue, int max = int.MaxValue)
        {
            int choice;
            do
            {
                try
                {
                    choice = int.Parse(Console.ReadLine());
                }
                catch
                {
                    choice = min - 1;
                    Console.Write("Invalid Choice Try Again: ");
                }
            } while (choice > max || choice < min); //while the input isnt within the set max and min or a valid int get a new into
            return choice;
        }

        public static double GetDouble(double min = double.MinValue, double max = double.MaxValue)
        {
            double choice;
            do
            {
                try
                {
                    choice = double.Parse(Console.ReadLine());
                }
                catch
                {
                    choice = min - 1;
                }
            } while (choice > max || choice < min); //while the input isnt within the set max and min or a valid double get a new into
            return choice;
        }

        static Neural_Network LoadAIFromFile()
        {
            Neural_Network AI;
            string path;
            do
            {
                Console.Write("Whats the path of your previous network (drag and drop file onto console): ");
                path = Console.ReadLine();
                try
                {
                    AI = Neural_Network.deserialiselaod(path);
                }
                catch
                {
                    Console.WriteLine("Load Failed! path might be wrong. Please try again.");
                    AI = null;
                }
            } while (AI == null);   //keeps trying to find a BIN file that can be deserialised and cast to class Neural_Network
            return AI;
        }

        [STAThread]
        static void Main(string[] args)
        {
            //initialising critical variables
            MNISTCON Data = null;
            MNISTCON DataT = null;
            Neural_Network AI = null;
            int firstLayerInputs = 0;
            Matrix<double> X;
            Matrix<double> yOH;
            Vector<double> yS;
            Matrix<double> Xt;
            Matrix<double> yOHt;
            Vector<double> ySt;

            //Gets which dataset you want to train on
            Console.Write("======= Which DataSet Would You Like To Use =======\n" +
                          "     1. MNIST (0-9)\n" +
                          ">>> "
                          );
            switch (GetInt(1, 1))
            {
                case 1:
                    firstLayerInputs = 784;
                    Data = new MNISTCON("Train");
                    DataT = new MNISTCON("");   //loads the dataset
                    break;
                default:
                    Console.Write("Not a DataSet");
                    break;
            }
            switch (displayMenu())
            {
                case 1:
                    //initialises Network with user layer count
                    Console.Write("How Many Layers? >>> ");
                    int hiddenLayers = GetInt();
                    AI = new Neural_Network(hiddenLayers, firstLayerInputs);
                    break;
                case 2:
                    AI = LoadAIFromFile();
                    break;
                default:
                    Console.WriteLine("Not an Option");
                    break;
            }

            //setting how long the user wants to train for
            Console.Write("How many iterations of the training dataset do you wish to do: ");
            int iterations = GetInt(min: 0);
            if (iterations != 0)
            {
                Console.Write("How images many per batch (0-10000): ");
                int batchsize = GetInt(1, 10000);
                Console.Clear();
                for (int i = 0; i < iterations; i++)
                {
                    for (int j = 0; j < 60000 / batchsize; j++)
                    {
                        (X, yOH, yS) = DigitImageToInput(Data.images.Skip(j * batchsize).Take(batchsize).ToList());
                        AI.forward(X, yOH);
                        if ((j + 1) % 20 == 0)
                        {
                            AI.displayLoss(i, j + 1, yS);
                        }
                        AI.backward(yS);
                    }
                }
            }
            //Does a forward pass of the whole Test dataset which it hasn't seen before
            (X, yOH, yS) = DigitImageToInput(DataT.images.Take(60000).ToList());
            AI.forward(X, yOH);

            //displays accuracy
            Console.WriteLine("Using Data the network hasn't seen before the Accurcy is: {0}%", AI.accuracy(yS.ToRowMatrix(), AI.activation_final.output) * 100);
            System.Threading.Thread.Sleep(3000);

            //if user is holding escape and space at the end seconds/ after the training it will exit straight away
            while (!Keyboard.IsKeyDown(Key.Escape))
            {
                //getting index of test dataset user wishes to see
                Console.Clear();
                Console.WriteLine("<<<Hold Escape and Space To Save and Exit>>>");
                Console.Write("Please choose an index of the Test Dataset (0-10000) >>> ");
                int setNumber = GetInt(0, 10000);

                DigitImage testImg = DataT.images[setNumber];

                //writes image as ASCII art and opens default image viewer with scaled bitmap image
                Console.WriteLine("ASCII representation on console: \n" + testImg.ToString());
                displayImage(testImg);


                //forwards the network on chosen number
                (Xt, yOHt, ySt) = DigitImageToInput(new List<DigitImage> { DataT.images[setNumber] });
                AI.Loss = 0;
                AI.forward(Xt, yOHt);

                //gets loss for that value and how confident the network is for each output

                Console.WriteLine("LOSS:{0}", AI.Loss);
                Console.WriteLine("ACC:{0}", accuracy(ySt.ToRowMatrix(), AI.activation_final.output));

                foreach (var item in AI.activation_final.output.EnumerateIndexed())
                {
                    Console.WriteLine("{0}: {1:0.00}%", item.Item2, item.Item3 * 100);
                }
                Console.WriteLine("<<<Press Space to do another number>>>");
                //waits until the keyboard is pressed
                while (!Keyboard.IsKeyDown(Key.Space))
                {
                }
                System.Threading.Thread.Sleep(50);
            }


            Neural_Network.serialisesave(AI);
        }
        /// <summary>
        /// saves image as a bmp number.bmp and opens defualt image
        /// </summary>
        /// <param name="number"></param>
        public static void displayImage(DigitImage number)
        {
            Bitmap bmp = new Bitmap(28, 28);

            for (int x = 0; x < 28; x++)
            {
                for (int y = 0; y < 28; y++)
                {
                    byte greyscaleColour = number.pixels[y][x];
                    bmp.SetPixel(x, y, Color.FromArgb(greyscaleColour, greyscaleColour, greyscaleColour));
                }
            }

            Bitmap bmpScaled = new Bitmap(bmp, 560, 560);

            bmpScaled.Save("number.bmp", ImageFormat.Bmp);
            Process image = Process.Start("explorer.exe", "number.bmp");
        }

        static void shuffle(List<DigitImage> data)  //shuffles images in a list
        {
            Random r = new Random();
            for (int i = 0; i < data.Count(); i++)
            {
                int temp2 = r.Next(i, data.Count() - 1);
                var temp = data[i];
                data[i] = data[temp2];
                data[temp2] = temp;
            }
        }

        static double accuracy(Matrix<double> target, Matrix<double> softmax)
        {
            double count = 0;
            double accuracy = 0;

            if (target.RowCount == 1) foreach (var itemV in softmax.EnumerateRowsIndexed()) if (itemV.Item2.MaximumIndex() == target[0, itemV.Item1]) count++;  //goes though each row using one hot to match
                                                                                                                                                                //where the network has guessed highest and compares to actual answer to see if its right
                                                                                                                                                                //adds to count value

                    else if (target.RowCount != 1) foreach (var itemM in softmax.EnumerateRowsIndexed()) if (itemM.Item2.MaximumIndex() == target.Row(itemM.Item1).MaximumIndex()) count++;
            accuracy = count / softmax.RowCount;

            return accuracy;
        }



    }

    [Serializable]
    class Neural_Network
    {
        double loss = 0;
        public double Loss { get => loss; set { loss = value; } }
        double data_loss = 0;
        double regularization_loss = 0;
        List<Layer_Dense> dense_layers = new List<Layer_Dense>();
        List<Activation_ReLU> activation_layers = new List<Activation_ReLU>();
        public Activation_Softmax_Loss_CategoricalCrossentropy activation_final = new Activation_Softmax_Loss_CategoricalCrossentropy();
        Optimizer optimizer;

        public Neural_Network(int layers, int firstLayerInputs)
        {
            chooseOptimizer(ref optimizer);
            optimizer.pre_update_params();
            int inputs = firstLayerInputs;
            for (int i = 0; i < layers; i++)
            {
                inputs = createDenseLayer(inputs, i + 1, layers);
                activation_layers.Add(new Activation_ReLU());   //creates dense layer list and ReLu List using output count of last layer as input for next layer
            }
            activation_layers.RemoveAt(0); //removes extra ReLu
        }

        static public void serialisesave(Neural_Network AI)
        {
            Stream stream = null;
            bool success = false;
            IFormatter formatter = new BinaryFormatter();
            do
            {
                try
                {
                    Console.Write("Path for saved file >>> ");
                    string path = Console.ReadLine();
                    stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);    //using serialisation to save to file to project folder
                    formatter.Serialize(stream, AI);
                    success = true;

                }
                catch
                {
                    Console.WriteLine("Path invalid try again");
                    success = false;
                }
            } while (!success);
            stream.Close();
        }

        static public Neural_Network deserialiselaod(string path)
        {
            IFormatter formatter = new BinaryFormatter();
            Stream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read); //use path and deserialise neural network
            Neural_Network obj = (Neural_Network)formatter.Deserialize(stream);
            stream.Close();
            return obj; //returns deserialised neural network
        }

        //initialses dense layer and adds to the list
        int createDenseLayer(int inputs, int layerNum, int totalLayers)
        {
            int outputs = 0;
            if (layerNum != totalLayers)
            {
                Console.Write("How many outputs for Layer {0} (0-1000): ", layerNum);
                outputs = Program.GetInt(0, 1000);
            }
            else
            {
                outputs = 10;
            }
            dense_layers.Add(new Layer_Dense(inputs, outputs, weight_regularizer_l2: 0.0005, bias_regularizer_l2: 0.0005));
            return outputs;

        }

        /// <summary>
        /// Goes through each layer and uses the forward function to pass the inputs forward 
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="y_OH"></param>
        public void forward(Matrix<double> inputs, Matrix<double> y_OH)
        {
            dense_layers[0].forward(inputs);
            for (int i = 1; i < dense_layers.Count; i++)
            {
                activation_layers[i - 1].forward(dense_layers[i - 1].output);
                dense_layers[i].forward(activation_layers[i - 1].outputs);
            }
            //finds the data_loss
            data_loss = activation_final.forward(dense_layers.Last().output, y_OH);
            regularization_loss = 0;
            foreach (var layer in dense_layers)
            {
                //finds regularization loss
                regularization_loss += activation_final.loss.CrossEntropy.regularization_loss(layer);
            }
            //updates the total loss
            loss += data_loss + regularization_loss;
        }

        public void backward(Vector<double> y_S)
        {
            //using back propogation to find the dvalues of each values and the dinputs
            activation_final.backward(activation_final.output, y_S);
            dense_layers.Last().backward(activation_final.dinputs);
            for (int i = dense_layers.Count - 2; i >= 0; i--)
            {
                activation_layers[i].backwards(dense_layers[i + 1].dinputs);
                dense_layers[i].backward(activation_layers[i].dinputs);
            }

            optimizer.pre_update_params();
            foreach (var layer in dense_layers)
            {
                //updates the weights of each layer
                optimizer.update_params(layer);
            }
            optimizer.post_update_params();
        }

        public void displayLoss(int i, int j, Vector<double> yS)
        {
            //displays information about the networks progression
            Console.WriteLine($"Iter: {i + 1:0000}," +
                                $" epoch: {j:0000}," +
                                $" acc: {accuracy(yS.ToRowMatrix(), activation_final.output):0.000}," +
                                $" loss: {loss / 20:0.000}," +
                                $" (data_loss: {data_loss:0.000}," +
                                $" reg_loss: {regularization_loss:0.000})," +
                                $" lr: {optimizer.GetLearningRate:0.00000}");
            //resets the average loss for each 20 batches
            loss = 0;
        }

        public double accuracy(Matrix<double> target, Matrix<double> softmax)
        {
            double count = 0;
            double accuracy = 0;
            //compares the strongest output to each label of the true values and finds a percentage of which ones were correct
            if (target.RowCount == 1) foreach (var itemV in softmax.EnumerateRowsIndexed()) if (itemV.Item2.MaximumIndex() == target[0, itemV.Item1]) count++;

                    else if (target.RowCount != 1) foreach (var itemM in softmax.EnumerateRowsIndexed()) if (itemM.Item2.MaximumIndex() == target.Row(itemM.Item1).MaximumIndex()) count++;
            accuracy = count / softmax.RowCount;

            return accuracy;
        }

        void chooseOptimizer(ref Optimizer opt)
        {
            //displays menu
            Console.Write("Which Optimizer do you want? \n" +
                          "     1. Adam \n" +
                          "     2. Adagrad \n" +
                          "     3. RMSprop \n" +
                          "     4. SGD \n" +
                          ">>> ");
            //gets user input on the common parameters
            int choice = Program.GetInt(1, 4);
            Console.Write("Learning Rate? >>> ");
            double learningrate = Program.GetDouble();
            Console.Write("Decay? >>> ");
            double Decay = Program.GetDouble();

            //depending on what optimizer they choose there will be different perameters and initializes the referenced optimizer
            switch (choice)
            {
                case 1:
                    Console.Write("Epsilon? >>> ");
                    double Epsilon = Program.GetDouble();
                    Console.Write("Beta1? >>> ");
                    double Beta1 = Program.GetDouble();
                    Console.Write("Beta2? >>> ");
                    double Beta2 = Program.GetDouble();
                    opt = new Optimizer_Adam(learningrate, Decay, Epsilon, Beta1, Beta2);
                    break;
                case 2:
                    Console.Write("Epsilon? >>> ");
                    Epsilon = Program.GetDouble();
                    opt = new Optimizer_Adagrad(learningrate, Decay, Epsilon);
                    break;
                case 3:
                    Console.Write("Epsilon? >>> ");
                    Epsilon = Program.GetDouble();
                    Console.Write("Rho? >>> ");
                    double Rho = Program.GetDouble();
                    opt = new Optimizer_RMSprop(learningrate, Decay, Epsilon, Rho);
                    break;
                case 4:
                    Console.Write("Momentum? >>> ");
                    double Momentum = Program.GetDouble();
                    opt = new Optimizer_SGD(learningrate, Decay, Momentum);
                    break;
            }
        }
    }

    [Serializable]
    class Optimizer
    {
        protected double Current_learning_rate;
        protected double learning_rate;
        protected double iterations = 0;
        public double GetLearningRate { get => Current_learning_rate; }
        protected double decay;
        public void pre_update_params()
        {
            //decays the learning rate to allow for smaller adjustments after a long time of learning
            if (decay > 0)
            {
                Current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
            }
        }

        public virtual void update_params(Layer_Dense layer) { Console.WriteLine("NOT WORKING"); }

        public void post_update_params()
        {
            iterations++;
        }

        //base class for which the actual optimizers override the functions allowing for polymorphism in the Neural_Network class
    }

    [Serializable]
    class Activation_Softmax_Loss_CategoricalCrossentropy
    {
        public Matrix<double> output;
        public Matrix<double> dinputs;
        public Activation_Softmax activation = new Activation_Softmax();
        public Loss loss = new Loss();

        public double forward(Matrix<double> inputs, Matrix<double> y_true)
        {
            //uses forward on the activation function
            activation.forward(inputs);
            output = activation.Probabilities;
            //calculates loss
            return loss.calculate(output, y_true);
        }

        public void backward(Matrix<double> dvalues, Vector<double> y_true)
        {
            //using backpropogation to find dinputs
            dinputs = dvalues.Clone();
            foreach (var item in dinputs.EnumerateRowsIndexed())
            {
                item.Item2[(int)y_true[item.Item1]] -= 1;
                dinputs.SetRow(item.Item1, item.Item2.Divide(dvalues.RowCount));
            }
        }
    }

    [Serializable]
    class Optimizer_Adam : Optimizer
    {
        double epsilon;
        double beta_1;
        double beta_2;
        public Optimizer_Adam(double learning_rate = 0.001, double decay = 0, double epsilon = 0.0000001, double beta_1 = 0.9, double beta_2 = 0.999)
        {
            //initialises parameters of the Adam Optimizer
            this.learning_rate = learning_rate;
            Current_learning_rate = learning_rate;
            this.decay = decay;
            this.epsilon = epsilon;
            this.beta_1 = beta_1;
            this.beta_2 = beta_2;
        }

        public override void update_params(Layer_Dense layer)
        {
            //for each layer update weight momentum and cache then update the weights themself
            layer.weight_Momentum = beta_1 * layer.weight_Momentum + (1 - beta_1) * layer.dweights;
            layer.bias_Momentum = beta_1 * layer.bias_Momentum + (1 - beta_1) * layer.dbiases;

            var weight_momentums_corrected = layer.weight_Momentum / (1 - Math.Pow(beta_1, iterations + 1));
            var bias_momentums_corrected = layer.bias_Momentum / (1 - Math.Pow(beta_1, iterations + 1));

            layer.weight_Cache = beta_2 * layer.weight_Cache + (1 - beta_2) * layer.dweights.PointwisePower(2);
            layer.bias_Cache = beta_2 * layer.bias_Cache + (1 - beta_2) * layer.dbiases.PointwisePower(2);

            var weight_cache_corrected = layer.weight_Cache / (1 - Math.Pow(beta_2, (iterations + 1)));
            var bias_cache_corrected = layer.bias_Cache / (1 - Math.Pow(beta_2, (iterations + 1)));


            layer.weights += -Current_learning_rate * weight_momentums_corrected.PointwiseDivide((weight_cache_corrected.PointwiseSqrt() + epsilon));
            layer.biases += -Current_learning_rate * bias_momentums_corrected.PointwiseDivide((bias_cache_corrected.PointwiseSqrt() + epsilon));
        }
    }
    class Optimizer_Adagrad : Optimizer
    {
        double epsilon;
        public Optimizer_Adagrad(double learning_rate = 1, double decay = 0, double epsilon = 0.0000001)
        {
            //initialises parameters of the Adagrad Optimizer
            this.learning_rate = learning_rate;
            Current_learning_rate = learning_rate;
            this.decay = decay;
            this.epsilon = epsilon;
        }

        /// <summary>
        /// updates weights cache for that layer then the weights themselves 
        /// </summary>
        /// <param name="layer"></param>
        public override void update_params(Layer_Dense layer)
        {
            layer.weight_Cache += layer.dweights.PointwisePower(2);
            layer.bias_Cache += layer.dbiases.PointwisePower(2);

            layer.weights += -Current_learning_rate * layer.dweights.PointwiseDivide(layer.weight_Cache.PointwiseSqrt().Add(epsilon));
            layer.biases += -Current_learning_rate * layer.dbiases.PointwiseDivide(layer.bias_Cache.PointwiseSqrt().Add(epsilon));
        }
    }
    class Optimizer_RMSprop : Optimizer
    {
        double epsilon;
        double rho;
        public Optimizer_RMSprop(double learning_rate = 0.001, double decay = 0, double epsilon = 0.0000001, double rho = 0.9)
        {
            //initialises parameters of the RMSprop Optimizer
            this.learning_rate = learning_rate;
            Current_learning_rate = learning_rate;
            this.decay = decay;
            this.epsilon = epsilon;
            this.rho = rho;
        }


        public override void update_params(Layer_Dense layer)
        {
            //updates weight/bias cache and then the weighrs/biases themselves
            layer.weight_Cache = rho * layer.weight_Cache + (1 - rho) * layer.dweights.PointwisePower(2);
            layer.bias_Cache = rho * layer.bias_Cache + (1 - rho) * layer.dbiases.PointwisePower(2);

            layer.weights += -Current_learning_rate * layer.dweights.PointwiseDivide(layer.weight_Cache.PointwiseSqrt().Add(epsilon));
            layer.biases += -Current_learning_rate * layer.dbiases.PointwiseDivide(layer.bias_Cache.PointwiseSqrt().Add(epsilon));
        }

    }
    class Optimizer_SGD : Optimizer
    {
        double momentum;
        public Optimizer_SGD(double learning_rate = 1, double decay = 0, double momentum = 0)
        {
            //initialises the parameters for the SGD optimizer
            this.learning_rate = learning_rate;
            Current_learning_rate = learning_rate;
            this.decay = decay;
            this.momentum = momentum;
        }

        public override void update_params(Layer_Dense layer)
        {
            //uses momentum to update weights and biases and adds them to the weights and biases
            Matrix<double> weight_updates;
            Vector<double> bias_updates;
            if (momentum > 0)
            {
                weight_updates = (momentum * layer.weight_Momentum) - (Current_learning_rate * layer.dweights);
                layer.weight_Momentum = weight_updates;
                bias_updates = (momentum * layer.bias_Momentum) - (Current_learning_rate * layer.dbiases);
                layer.bias_Momentum = bias_updates;
            }
            else
            {
                weight_updates = -Current_learning_rate * layer.dweights;
                bias_updates = -Current_learning_rate * layer.dbiases;
            }

            layer.weights += weight_updates;
            layer.biases += bias_updates;
        }
    }

    interface CrossEntropy
    {
        //interfac for Loss and CatagoricalCrossentropy
        Vector<double> forward(Matrix<double> y_pred, Matrix<double> y_true);
        void backward(Matrix<double> dvalues, Matrix<double> y_true);
    }
    [Serializable]
    class Loss_CategoricalCrossentropy : CrossEntropy
    {
        public Matrix<double> dinputs;
        public Vector<double> forward(Matrix<double> y_pred, Matrix<double> y_true)
        {
            //returns the loss for each image in the batch as a vector
            var losses = CreateVector.Dense<double>(y_pred.RowCount);
            clip(y_pred, 0.0000001, 1 - 0.0000001);

            if (y_true.RowCount == 2)
                foreach (var index in y_true.EnumerateColumnsIndexed())
                    losses[index.Item1] = -Math.Log(y_pred[index.Item1, (int)index.Item2[0]]);

            else
                foreach (var index in y_true.EnumerateRowsIndexed())
                    losses[index.Item1] = -Math.Log(y_pred[index.Item1, Array.FindIndex(index.Item2.AsArray(), i => i == 1)]);

            return losses;
        }

        public double regularization_loss(Layer_Dense layer)
        {
            //calculates regularization loss
            double regularization_loss = 0;

            if (layer.weight_regularizer_l1 > 0) regularization_loss += layer.weight_regularizer_l1 * layer.weights.RowAbsoluteSums().Sum();

            if (layer.weight_regularizer_l2 > 0) regularization_loss += layer.weight_regularizer_l2 * layer.weights.PointwisePower(2).RowSums().Sum();


            if (layer.bias_regularizer_l1 > 0) regularization_loss += layer.bias_regularizer_l1 * layer.biases.PointwiseAbs().Sum();

            if (layer.bias_regularizer_l2 > 0) regularization_loss += layer.bias_regularizer_l2 * layer.biases.PointwisePower(2).Sum();

            return regularization_loss;
        }

        public void backward(Matrix<double> dvalues, Matrix<double> y_true)
        {
            //uses back propogation to find the dvalues of each images outputs
            dinputs = CreateMatrix.Dense<double>(dvalues.RowCount, dvalues.ColumnCount);
            foreach (var item in y_true.EnumerateIndexed()) dinputs[item.Item1, item.Item2] = (-item.Item3 / dvalues[item.Item1, item.Item2]) / dvalues.ColumnCount;
        }



        void clip(Matrix<double> input, double min, double max)
        {
            //clips all the values in the inputs to a min and max value
            foreach (var item in input.EnumerateIndexed())
            {
                if (item.Item3 < min) input[item.Item1, item.Item2] = min;

                else if (item.Item3 > max) input[item.Item1, item.Item2] = max;
            }
        }
    }

    [Serializable]
    class Loss : CrossEntropy
    {
        public Loss_CategoricalCrossentropy CrossEntropy = new Loss_CategoricalCrossentropy();
        public Vector<double> forward(Matrix<double> y_pred, Matrix<double> y_true) { return CrossEntropy.forward(y_pred, y_true); }
        public void backward(Matrix<double> dvalues, Matrix<double> y_true) { CrossEntropy.backward(dvalues, y_true); }

        public double calculate(Matrix<double> output, Matrix<double> y)
        {
            //calculates data loss
            Vector<double> sample_losses = forward(output, y);
            double data_loss = sample_losses.Average();
            return data_loss;
        }


    }

    [Serializable]
    class Layer_Dense
    {
        //declares all matices, vectors and params for the layer
        public Matrix<double> weights;
        public Matrix<double> dweights;
        public Matrix<double> weight_Momentum;
        public Matrix<double> weight_Cache;
        public Vector<double> biases;
        public Vector<double> dbiases;
        public Vector<double> bias_Momentum;
        public Vector<double> bias_Cache;
        public Matrix<double> output;
        public Matrix<double> inputs;
        public Matrix<double> dinputs;
        public double weight_regularizer_l1;
        public double weight_regularizer_l2;
        public double bias_regularizer_l1;
        public double bias_regularizer_l2;
        Matrix<double> dL1M;
        Vector<double> dL1V;
        public Layer_Dense(int inputs, int neurons, double weight_regularizer_l1 = 0, double weight_regularizer_l2 = 0, double bias_regularizer_l1 = 0, double bias_regularizer_l2 = 0)
        {
            //initialeses random weights and empty matrices for momentum and cache 
            weights = CreateMatrix.Random<double>(inputs, neurons, new Normal(0, 0.1));
            weight_Momentum = CreateMatrix.Dense<double>(inputs, neurons);
            weight_Cache = CreateMatrix.Dense<double>(inputs, neurons);
            biases = CreateVector.Dense<double>(neurons);
            bias_Momentum = CreateVector.Dense<double>(neurons);
            bias_Cache = CreateVector.Dense<double>(neurons);
            //initialises learning params
            this.weight_regularizer_l1 = weight_regularizer_l1;
            this.weight_regularizer_l2 = weight_regularizer_l2;
            this.bias_regularizer_l1 = bias_regularizer_l1;
            this.bias_regularizer_l2 = bias_regularizer_l2;
        }

        public void forward(Matrix<double> inputs)
        {
            //calculates outputs of the layer with matrix math
            this.inputs = inputs;
            output = inputs.Multiply(weights);
            foreach (var item in output.EnumerateRowsIndexed()) output.SetRow(item.Item1, item.Item2 + biases);
        }
        /// <summary>
        /// uses back propogation to calculate dinputs dweights and dbiases
        /// </summary>
        /// <param name="dvalues"></param>
        public void backward(Matrix<double> dvalues)
        {

            dweights = inputs.Transpose() * dvalues;
            dbiases = dvalues.ColumnSums();

            if (weight_regularizer_l1 > 0)
            {
                foreach (var item in weights.EnumerateIndexed())
                {
                    dL1M[item.Item1, item.Item2] = item.Item3.CompareTo(0) < 0 ? -1 : 1;
                }
                dweights += weight_regularizer_l1 * dL1M;
            }

            if (weight_regularizer_l2 > 0)
            {
                dweights += 2 * weight_regularizer_l2 * weights;
            }

            if (bias_regularizer_l1 > 0)
            {
                foreach (var item in biases.EnumerateIndexed())
                {
                    dL1V[item.Item1] = item.Item2.CompareTo(0) < 0 ? -1 : 1;
                }
                dbiases += 2 * bias_regularizer_l1 * dL1V;
            }


            if (bias_regularizer_l2 > 0)
            {
                dbiases += 2 * bias_regularizer_l2 * biases;
            }

            dinputs = dvalues * weights.Transpose();
        }

    }

    [Serializable]
    class Activation_ReLU
    {
        public Matrix<double> outputs;
        public Matrix<double> inputs;
        public Matrix<double> dinputs;
        public void forward(Matrix<double> inputs)
        {
            //all values less than zero are not included in the output
            this.inputs = inputs.Clone();
            outputs = CreateMatrix.Dense<double>(inputs.RowCount, inputs.ColumnCount);
            foreach (var item in inputs.EnumerateIndexed()) outputs[item.Item1, item.Item2] = item.Item3.CompareTo(0) == 1 ? item.Item3 : 0;
        }

        public void backwards(Matrix<double> dvalues)
        {
            //using inputs to find the outputs by comparing them to 0
            dinputs = dvalues.Clone();
            foreach (var item in inputs.EnumerateIndexed())
                dinputs[item.Item1, item.Item2] = item.Item3.CompareTo(0) == -1 ? 0 : dinputs[item.Item1, item.Item2];
        }
    }
    [Serializable]
    class Activation_Softmax
    {
        public Matrix<double> exp_Values;
        public Matrix<double> Probabilities;
        public Matrix<double> dinputs;
        public void forward(Matrix<double> inputs)
        {
            exp_Values = CreateMatrix.Dense<double>(inputs.RowCount, inputs.ColumnCount);

            Probabilities = CreateMatrix.Dense<double>(inputs.RowCount, inputs.ColumnCount);
            //normalises valus
            foreach (var item in inputs.EnumerateRowsIndexed())
                inputs.SetRow(item.Item1, item.Item2.Subtract(item.Item2.Max()));
            //raises e to the power of each value
            foreach (var item in inputs.EnumerateIndexed())
                exp_Values[item.Item1, item.Item2] = Math.Pow(Math.E, item.Item3);
            //finds probabilities for the confidence of each output
            foreach (var item in exp_Values.EnumerateIndexed())
                Probabilities[item.Item1, item.Item2] = exp_Values[item.Item1, item.Item2] / exp_Values.Row(item.Item1).Sum();
        }

        public void backward(Matrix<double> dvalues)
        {
            //uses back propogation to calculate the dinputs for each image
            dinputs = CreateMatrix.Dense<double>(Probabilities.RowCount, Probabilities.ColumnCount);
            foreach (var item in Probabilities.EnumerateRowsIndexed()) dinputs.SetRow(item.Item1, (CreateMatrix.DiagonalOfDiagonalVector(item.Item2) - (item.Item2.ToColumnMatrix() * item.Item2.ToRowMatrix())) * dvalues.Row(item.Item1));
        }
    }
}
