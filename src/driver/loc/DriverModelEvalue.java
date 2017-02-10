package driver.loc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSink;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


/*
 * ������Ҫ�����������ļ�������������ģ�ͣ�����.model�ļ�
 * <cuid,clusterID,Centerx,centery,wifi_entropy,ratio_present_day,avg_stay_time,ratio_daytime,r
 * atio_night,ratio_weekend,ratio_slot_0,ratio_slot_1,ratio_slot_2,ratio_slot_3,ratio_slot_4,
 * ratio_slot_5,ratio_slot_6,ratio_slot_7,label>
 * 
 */
public class DriverModelEvalue {

	public DriverModelEvalue() {
		// TODO Auto-generated constructor stub
	}
	
	
	public static Instances txtFileToInstances(String fileName){
		
		
		final Attribute CUID = new Attribute("cuid",(FastVector)null);//string
		
		//��������14������
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

		//1 ����ǩ,�����������
		FastVector labels = new FastVector();
		labels.addElement("H");
		labels.addElement("C");
		labels.addElement("O");		
		final Attribute LABEL = new Attribute("label",(FastVector)labels);

		
		FastVector atts; // 3.6 ʹ��FastVector 
		atts = new FastVector(16);
		atts.addElement(CUID);
		atts.addElement(WIFI_ENTROPY); 
		atts.addElement(PRESENT_DAY);// numberic����			
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
				
				Instance inst = new Instance(16);	
				
				inst.setValue(CUID,parts[0].trim());
				
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
				inst.setDataset(instances);
				instances.add(inst);	
				
			}

			
		}catch(Exception e){
			e.printStackTrace();
		}

		
		
		
		return instances;
		
	}
	
	

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub		

		if (args.length!=1) {
			System.out.println("java jar xx.jar testfilename");
			return;
		}
		
		String fileName = args[0];
		
		File file = new File(fileName);

		if(!file.exists()){
			System.out.println("file do not exists:"+file.getName());
			return;
		}
		
	
		try {
			
			//��ȡ�ļ�������ѵ������instances		
			Instances testSetOrgin = txtFileToInstances(fileName);	
			
			//��Ԥ�⼯�ϵ�һ�����ԣ�cuidȥ��
			String col2 = "1";
			Remove mv2 = new Remove();
			mv2.setAttributeIndices(col2);
			mv2.setInputFormat(testSetOrgin);			
			Instances testSet = Filter.useFilter(testSetOrgin, mv2);
			
			testSet.setClassIndex(testSet.numAttributes()-1);  //������𣬴�0��ʼ				
			
			//ʹ�÷���ѧϰ��ģ��				
			Object obj[]= null;		
			obj = SerializationHelper.readAll("./j48.model");
			Classifier tree = (Classifier)obj[0];
			Instances trainSet = (Instances) obj[1];
			
			double pred =-1;
			double[] dist = null;
			String preClass = null;
			String preClass2 = null;
			double real = -1;
			
			if(!testSet.equalHeaders(trainSet)){
				//{H,C,O}
				System.out.println("trainSet and testSet not compatible!");
			}
			
			long realHomeTotal =0;//��ʵ���мҵ�����
			HashSet<String> preHome = new HashSet<String>();
			HashSet<String> preHomeAndRight = new HashSet<String>();
			
			//HashMap<String, Double> preHomeAndRight = new HashMap<String, Double>();
			
			long realComTotal =0;//��ʵ���мҵ�����
			HashSet<String> preCom = new HashSet<String>();
			HashSet<String> preComAndRight = new HashSet<String>();
			
			System.out.println("begin to statistic data!");
			
			for(int i=0;i<testSet.numInstances();i++){
				
				pred = tree.classifyInstance(testSet.instance(i));  //�����Լ��е�ʵ��Ԥ��ʲô�������ֵ
				dist = tree.distributionForInstance(testSet.instance(i));
				
				preClass = testSet.classAttribute().value((int)pred);				
				preClass2 = trainSet.classAttribute().value((int)pred);
				
				real = testSet.instance(i).classValue();//ʵ�ʵ����ֵ
				
				String cuid = testSetOrgin.instance(i).toString(0);
				
				if(real == 0){ //��ʵ�ı�ǩ����ʶΪ�� 0
					realHomeTotal++;
				}
				
				if(pred == 0){//������¼Ԥ��Ϊ�ң���¼���û�
					preHome.add(cuid);
				}
				
				if((pred == 0)&&(pred == real)){//������¼Ԥ��Ϊ�ң���¼���û�
					preHomeAndRight.add(cuid);
				}
				
				
				if(real == 1){ //��ʵ�ı�ǩ����ʶΪ��˾
					realComTotal++;
				}
				
				if(pred == 1){//������¼Ԥ��Ϊ�ң���¼���û�
					preCom.add(cuid);
				}
				
				if((pred == 1)&&(pred == real)){//������¼Ԥ��Ϊ�ң���¼���û�
					preComAndRight.add(cuid);
				}		
				
				
//				System.out.println("instance:"+testSet.instance(i));
//				System.out.println("instance:"+testSetOrgin.instance(i));
//				System.out.println("pre:dist:preclass: "+pred+"\t"+preClass+"\t"+preClass2+"\t"+real);
//				System.out.println(Utils.arrayToString(dist));
				
				
			}
			
			System.out.println("preHOMEandRight:preHOME:HOME:recall:accur  "
					+ preHomeAndRight.size() + "\t" + preHome.size() + "\t"
					+ realHomeTotal + "\t" + (double) preHomeAndRight.size()
					/ realHomeTotal + "\t" + (double) preHomeAndRight.size()
					/ preHome.size());

			System.out.println("preCOMandRight:preCOM:COM:recall:accur "
					+ preComAndRight.size() + "\t" + preCom.size() + "\t"
					+ realComTotal + "\t" + (double) preComAndRight.size()
					/ realComTotal + "\t" + (double) preComAndRight.size()
					/ preCom.size());		
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		

	
		
		

	}

}
