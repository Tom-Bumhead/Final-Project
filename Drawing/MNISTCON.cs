using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetworkLearning
{
    class MNISTCON
    {
        public List<DigitImage> images = new List<DigitImage>();
        public MNISTCON(string ToT)
        {
            //creates the stream to train images and labels files
            FileStream ifsLabels = ToT == "Train" ? new FileStream("train-labels.idx1-ubyte", FileMode.Open) : new FileStream("t10k-labels.idx1-ubyte", FileMode.Open);
            FileStream ifsImages = ToT == "Train" ? new FileStream("train-images.idx3-ubyte", FileMode.Open) : new FileStream("t10k-images.idx3-ubyte", FileMode.Open);
            int iterations = ToT == "Train" ? 60000 : 10000;

            //creates binary readers for each file
            BinaryReader brLabels = new BinaryReader(ifsLabels);
            BinaryReader brImages = new BinaryReader(ifsImages);

            //meta data
            int magic1 = brImages.ReadInt32();
            int numImages = brImages.ReadInt32();
            int numRows = brImages.ReadInt32();
            int numCols = brImages.ReadInt32();

            int magic2 = brLabels.ReadInt32();
            int numLabels = brLabels.ReadInt32();

            //creates 2D pixel array
            byte[][] pixels = new byte[28][];
            for (int i = 0; i < pixels.Length; ++i)
                pixels[i] = new byte[28];

            //reads each image as greyscale image
            for (int di = 0; di < iterations; ++di)
            {
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        byte b = brImages.ReadByte();
                        pixels[i][j] = b;
                    }
                }
                //reads label
                byte lbl = brLabels.ReadByte();

                //creates DigitImage from label and pixel data
                DigitImage dImage =
                  new DigitImage(pixels, lbl);

                //adds to list of DigitImage
                images.Add(dImage);
            }

            //closes all streams
            ifsImages.Close();
            brImages.Close();
            ifsLabels.Close();
            brLabels.Close();
        }
    }


    public class DigitImage
    {
        public byte[][] pixels;
        public byte label;

        public DigitImage(byte[][] pixels,
          byte label)
        {
            //initialises pixels and sets pixel data
            this.pixels = new byte[28][];
            for (int i = 0; i < this.pixels.Length; ++i)
                this.pixels[i] = new byte[28];

            for (int i = 0; i < 28; ++i)
                for (int j = 0; j < 28; ++j)
                    this.pixels[i][j] = pixels[i][j];

            //sets label
            this.label = label;
        }

        public override string ToString()
        {
            //creates ASCII representation of image
            string s = "";
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    if (this.pixels[i][j] == 0)
                        s += " "; // white
                    else if (this.pixels[i][j] == 255)
                        s += "O"; // black
                    else
                        s += "."; // gray
                }
                s += "\n";
            }
            //adds label to end of string
            s += label.ToString();
            return s;
        }

    }


}

