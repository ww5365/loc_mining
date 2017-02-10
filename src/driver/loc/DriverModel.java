package driver.loc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSink;


/*
 * 本类主要用来对特征文件，来构建分类模型，生成.model文件
 * <cuid,clusterID,Centerx,centery,wifi_entropy,ratio_present_day,avg_stay_time,ratio_daytime,r
 * atio_night,ratio_weekend,ratio_slot_0,ratio_slot_1,ratio_slot_2,ratio_slot_3,ratio_slot_4,
 * ratio_slot_5,ratio_slot_6,ratio_slot_7,label>
 * 
 */
public class DriverModel {

	public DriverModel() {
		// TODO Auto-generated constructor stub
	}
	
	
	public static Instances txtFileToInstances(String fileName,double homeWeight,double comWeight,double otherWeight){
		
		
		final Attribute CUID = new Attribute("cuid",(FastVector)null);//string
		
		//共计以下14个特征
		final Attribute WIFI_ENTROPY = new Attribute("wifi_entropy");
		final Attribute PRESENT_DAY = new Attribute("present_day");
		final Attribute AVG_STAY_TIME = new Attribute("avg_stay_time");
		
		final Attribute DAY_TIME = new Attribute("day_time");
		final Attribute NIGHT_TIME = new Attribute("night_time");
		final Attribute WEEKEND = new Attribute("weekend");
		
		final Attribute SLOT_TIME_0 = new Attribute("slot_time_0");
		final Attribute SLOT_TIME_1 = new Attribute("slot_time_1");
		final Attribute SLOT_TIME_2 = new Attribute("slot_time_2");
		final Attribute SLOT_TIME_3 = new Attribute("slot_time_3");
		final Attribute SLOT_TIME_4 = new Attribute("slot_time_4");
		final Attribute SLOT_TIME_5 = new Attribute("slot_time_5");
		final Attribute SLOT_TIME_6 = new Attribute("slot_time_6");
		final Attribute SLOT_TIME_7 = new Attribute("slot_time_7");

		//1 个标签,标称类型数据
		FastVector labels = new FastVector();
		labels.addElement("H");
		labels.addElement("C");
		labels.addElement("O");		
		final Attribute LABEL = new Attribute("label",(FastVector)labels);

		
		FastVector atts; // 3.6 使用FastVector 
		atts = new FastVector(15);
		
		atts.addElement(WIFI_ENTROPY); 
		atts.addElement(PRESENT_DAY);// numberic类型			
		atts.addElement(AVG_STAY_TIME);
		
		atts.addElement(DAY_TIME);
		atts.addElement(NIGHT_TIME);
		atts.addElement(WEEKEND);
		
		atts.addElement(SLOT_TIME_0);
		atts.addElement(SLOT_TIME_1);			
		atts.addElement(SLOT_TIME_2);
		atts.addElement(SLOT_TIME_3);
		atts.addElement(SLOT_TIME_4);
		atts.addElement(SLOT_TIME_5);
		atts.addElement(SLOT_TIME_6);
		atts.addElement(SLOT_TIME_7);
		
		atts.addElement(LABEL);
		
		Instances instances = new Instances("driverModel",atts,0);	
		
		try{			
			InputStreamReader read = new InputStreamReader(new FileInputStream(fileName));		
			BufferedReader bufReader = new BufferedReader(read);		
			String line = null;		
			String[] parts;	
			
			
			while((line= bufReader.readLine())!=null){
				
				parts = line.split("\t");
				
				if(19!=parts.length){
					continue;
				}
				
				//System.out.println("line len: "+parts.length);
				
				//System.out.println(line);
				
				Instance inst = new Instance(15);				
				
				inst.setValue(WIFI_ENTROPY,Double.parseDouble(parts[4]));
				inst.setValue(PRESENT_DAY,Double.parseDouble(parts[5]));
				inst.setValue(AVG_STAY_TIME,Double.parseDouble(parts[6]));
				
				inst.setValue(DAY_TIME,Double.parseDouble(parts[7]));
				inst.setValue(NIGHT_TIME,Double.parseDouble(parts[8]));
				inst.setValue(WEEKEND,Double.parseDouble(parts[9]));
				
				inst.setValue(SLOT_TIME_0,Double.parseDouble(parts[10]));
				inst.setValue(SLOT_TIME_1,Double.parseDouble(parts[11]));
				inst.setValue(SLOT_TIME_2,Double.parseDouble(parts[12]));
				inst.setValue(SLOT_TIME_3,Double.parseDouble(parts[13]));
				inst.setValue(SLOT_TIME_4,Double.parseDouble(parts[14]));
				inst.setValue(SLOT_TIME_5,Double.parseDouble(parts[15]));
				inst.setValue(SLOT_TIME_6,Double.parseDouble(parts[16]));
				inst.setValue(SLOT_TIME_7,Double.parseDouble(parts[17]));
				inst.setValue(LABEL,parts[18]);				
				
				if("H".equalsIgnoreCase(parts[18])){
					inst.setWeight(homeWeight);
				}else if("C".equalsIgnoreCase(parts[18])){
					inst.setWeight(comWeight);
				}else{
					inst.setWeight(otherWeight);
				}

				inst.setDataset(instances);
				
				instances.add(inst);	
				
			}

			
		}catch(Exception e){
			e.printStackTrace();
		}

		//怎么处理生成Instances？平衡样本比列 (使得h,c,o比列平衡 )
		
		
		
		
		
		return instances;
		
	}
	
	

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub		

		if (args.length!=4) {			
			
			/*    
			 *    此功能是：单机跑的，生成车主的model文件
			 *    
			 *    输入参数：特征文件，DriverClusterFeature.java 程序生成
			 *    输入文件的格式：
			 *    (clusterID, <Centerx,centery,wifi_entropy,ratio_present_day,avg_stay_time,
			 *    ratio_daytime,ratio_night,ratio_weekend,ratio_slot_0,ratio_slot_1,ratio_slot_2,
			 *    ratio_slot_3,ratio_slot_4,ratio_slot_5,ratio_slot_6,ratio_slot_7,label>)
			 */
			
			System.out.println("java jar xx.jar filename h c o");
			return;
		}
		
		String fileName = args[0];
		
		File file = new File(fileName);

		double hweight = Double.parseDouble(args[1]);
		double cweight = Double.parseDouble(args[2]);
		double oweight = Double.parseDouble(args[3]);
		
		if(!file.exists()){
			System.out.println("file do not exists:"+file.getName());
			return;
		}
		
		//读取文件，构建训练集的instances		
		Instances trainSet = txtFileToInstances(fileName,hweight,cweight,oweight);
		
		trainSet.setClassIndex(trainSet.numAttributes()-1);  //设置类别，从0开始
		
		String arffName = "./test.arff";
		File arffFile = new File(arffName);
		if (!arffFile.exists()) {
			arffFile.createNewFile();// 没有文件,新建
			System.out.println("create new file:"+arffFile);
		} 
		DataSink.write(arffName, trainSet);	
		
		
		for(int i=0;i<trainSet.numInstances();i++){
			Instance instance = trainSet.instance(i);
			
			System.out.println(instance.weight());
		}
		
		System.out.println("the instacne num:"+ trainSet.numInstances());
		
		
		//构建分类学习的模型		
		J48 tree =  new J48();		
		String []option = new String[1];
		option[0] = "-U";		
		tree.setOptions(option);
		tree.buildClassifier(trainSet);
		
		//System.out.println(trainSet.classAttribute());		
		//System.out.println(tree);
		
		//序列化结果:保存学习模型和数据集的头信息
		Instances header = new Instances(trainSet,0);		
		SerializationHelper.writeAll("./j48.model",new Object[]{tree,header});
		
		System.out.println("序列化分类器及头信息成功!");		
		
	
		
		

	}

}
