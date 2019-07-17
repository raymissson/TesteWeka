/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package testeweka;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author raymison.ramos
 */
public class TesteWeka {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        DataSource ds = new DataSource("src/testeweka/teste.arff");
        Instances ins = ds.getDataSet();
        //System.out.println(ins.toString());
        
        ins.setClassIndex(1);
        
        NaiveBayes nb = new NaiveBayes();
        
        nb.buildClassifier(ins);
        
        Instance novo = new DenseInstance(2);
        
        novo.setDataset(ins);
        
        novo.setValue(0, "QUARTOS");
        
        double probabilidade[] = nb.distributionForInstance(novo);
        
        System.out.println("Probilidade Q:"+probabilidade[16]);
    }
    
}
