package driver.loc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;


import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import weka.classifiers.Classifier;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.DBSCAN;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


public class UserClusterFeatureV3 {


	/**
	 * @param args
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		
		JobConf job = new JobConf();
		job.setJarByClass(UserClusterFeatureV3.class);
		
		job.setMapOutputKeyClass(Text.class);//�趨map�����keyʹ���ı��࣬������ʽ
		job.setMapOutputValueClass(Text.class);
		
		
		job.setInputFormat(TextInputFormat.class);//һ���е���������		
		job.setOutputFormat(TextOutputFormat.class);
		
		job.setMapperClass(MapUserCluster.class);
		job.setReducerClass(RedUserStat.class);
		
		job.set("mapred.job.priority", "VERY_HIGH");
		job.set("mapred.job.queue.name", "map-international");		
		
		job.setNumReduceTasks(400);
		
		//����hadoopƽ̨�ϵ������ļ�,ƥ�䣬������
		
		//FileInputFormat.setInputPaths(job,"/app/map/map-international/wangwei/tmp/staypoi_*");
		FileInputFormat.setInputPaths(job,"/app/lbs/lbs-guiji/ident_traj/mark_i2c/201511*");

		Path outPath = new Path("/app/map/map-international/wangwei/tmp/user_feature_new");
		//Path outPath = new Path("/app/map/map-international/wangwei/tmp/user_feature_test");
		
		FileSystem fs = outPath.getFileSystem(job);
		
		if(fs.exists(outPath)){
			fs.delete(outPath,true); //����ļ����ڵ�����£�ɾ�����ļ���
		}
		
		FileOutputFormat.setOutputPath(job, outPath);
		
		//����ʹ��hadoopƽ̨����ʱ�ļ�
		
		job.set("mapred.create.symlink", "yes");
		job.set("mapred.cache.files", "/app/map/map-international/wangwei/tmp/model/j48.model#j48.model");

		
		//�������������Ĳ���		
		String[] otherArgs = new GenericOptionsParser(job, args).getRemainingArgs();	
		
		System.out.println("job created and args: "+otherArgs.length);
		
		DistributedCache.createSymlink(job);
		
		int n = JobClient.runJobReturnExitCode(job);
		if(n != 0 ){
			System.out.println("Job FAIL");
		}
		System.exit(0);
		
	}
	
	 
    
	static class MapUserCluster extends MapReduceBase  implements Mapper<LongWritable, Text, Text, Text> {
		
		public String epsion;
		public String minPoint;		
		public ArrayList<String> userPoiList;
		public String lastCuid;
		public boolean firstLineFlag;
		
		
		/*
		 * configure ���mapper�ĳ�ʼ������(non-Javadoc)
		 * 
		 */
		public void configure(JobConf context)  
		{
			try{   	
				
	    		System.out.println("Mapper configure success!!!");
			}
			catch(Exception e){
				e.printStackTrace();
				System.err.println("Mapper configure error!!");
			}
		}
		
		
		/*
		 * @see org.apache.hadoop.mapred.Mapper#map(java.lang.Object, java.lang.Object, org.apache.hadoop.mapred.OutputCollector, org.apache.hadoop.mapred.Reporter)
		 * @Override
		 */
		
		public void map(LongWritable key, Text line,OutputCollector<Text, Text> output, Reporter rep)throws IOException {
			

			String word = line.toString();
			
			//System.out.println("map:line "+word);
			
			String[] parts=word.split("\t");
			
			if(parts.length!=11){
				//�û��켣���ݣ���11���ֶΣ�cuid,timestamp,x,y,r,conn_wifi,is_valid_wifi,wifi_list,cuid_2,appsrc,staypoint_type
				return;				
			}
			
			
			String cuidStr = parts[0].trim();
			String xx = parts[2].trim();
			String yy = parts[3].trim();
			String createTime = parts[1].trim();
			String wifi = parts[5].trim();	
			String stayType = parts[10].trim();
			
			if("stay_d".equalsIgnoreCase(stayType)||"stay_w".equalsIgnoreCase(stayType)){				
				output.collect(new Text(cuidStr),new Text(xx+"\t"+yy+"\t"+createTime+"\t"+wifi));//�����<cuid,(x,y,time,wifi),()>					
			}
			
			//System.out.println("line:"+word);
			//System.out.println("haha:"+cuidStr+"\t"+xx+"\t"+yy+"\t"+createTime+"\t"+wifi);
  
		}
		
		
	
		
	}
	
	
	/*
	 * @reducer
	 */
	
	static class RedUserStat extends MapReduceBase implements Reducer<Text, Text, Text, Text>{
		
		public ArrayList<String> userPoiList;
		public ArrayList<String> userPoiAndCluIDList;		
		public HashMap<String, ArrayList<String>> statRes;
		public String epsion;
		public String minPoint;
		
		
		
		/*
		 * configure ���reducer�ĳ�ʼ������(non-Javadoc)
		 * 
		 */
		public void configure(JobConf context)  
		{
			try{	
				userPoiList = new ArrayList<String>();	
				userPoiAndCluIDList = new ArrayList<String>();
				statRes = new HashMap<String, ArrayList<String>>();
				
				epsion = "0.008";				
				minPoint = "5";
				
				//��ȡ���л����ݣ������л�
				Path path[]=null;			
				path=DistributedCache.getLocalCacheFiles(context);				
				
	    		System.out.println("reducer configure success!!!");
			}
			catch(Exception e){
				e.printStackTrace();
				System.err.println("reducer configure error!!");
			}
		}
		
		
		/*
		 * ÿ���û������࣬���ϴغţ����
		 */
		public void processUserInfo(ArrayList<String> userPoiList)throws IOException {
			
			if(userPoiList.size()<=0){
				
				System.out.println("userPoiList empty");
				return;
			}
			
			FastVector atts = new FastVector(2);
			atts.addElement(new Attribute("X"));//numberic����
			atts.addElement(new Attribute("Y"));			
			Instances userPoiInstances = new Instances("userloc",atts,0);	
			String[] parts;
			
			//�������ʹ�õ�instancesλ�õ�����
			userPoiInstances.delete();					
			for(int i=0;i<userPoiList.size();i++){					
				parts=userPoiList.get(i).split("\t");
				Instance inst = new Instance(2);
				String logitude = parts[0].trim();   //�����û��ľ���
				String latitude = parts[1].trim();	
				
				if(logitude==null){
					logitude = "0";
				}
				if(latitude==null){
					latitude = "0";
				}					
				inst.setValue(0,Double.parseDouble(logitude));//numberic ����
				inst.setValue(1,Double.parseDouble(latitude));					
				inst.setDataset(userPoiInstances);
				userPoiInstances.add(inst);					
			}			
			
			
			//������û��ľ�̬�켣������
			DBSCAN clusterer = new DBSCAN(); 
			clusterer.setEpsilon(Double.parseDouble(epsion));
			clusterer.setMinPoints(Integer.parseInt(minPoint));	
			try{
				clusterer.buildClusterer(userPoiInstances);					
			}catch(Exception e){
				e.printStackTrace();
			}				
			
			// ����
			ClusterEvaluation eval = new ClusterEvaluation();
			eval.setClusterer(clusterer);
			try{
				eval.evaluateClusterer(userPoiInstances);
			}catch(Exception e){
				e.printStackTrace();
			}
			
			//��ȡ���༯����ÿһ��ʵ���������Ĵغ�
			double []clusterAssign = eval.getClusterAssignments();
			
			//ÿ����¼һ��ʵ����ÿ��ʵ����һ���غ�
			for(int i=0;i<userPoiList.size();i++){					
				parts=userPoiList.get(i).split("\t");
				String logiTmp = parts[0].trim();
				String latiTmp = parts[1].trim();
				String timeTmp = parts[2].trim();
				String wifiTmp = parts[3].trim();
				String clusterTmp = Integer.toString((int)clusterAssign[i]);					
				//String clusterTmp = Integer.toString((int)0);
				//output.collect(new Text(cuidTmp), new Text(logiTmp+"\t"+latiTmp+"\t"+timeTmp+"\t"+wifiTmp+"\t"+clusterTmp));
				
				userPoiAndCluIDList.add(new String(logiTmp+"\t"+latiTmp+"\t"+timeTmp+"\t"+wifiTmp+"\t"+clusterTmp));
			
			}		
			
		}	
		
		/*
		 * ��������ֵ��Instance
		 */
		
		public  Instance makeInstance(String feature,Instances testSet){
			
			if(feature==null){
				return null;
			}
			String[] parts = feature.split("\t");
			
			if(parts.length!=14){
				
				//System.out.println("parts len: "+ parts.length);
				return null;
			}
			
			Instance testInst = new Instance(15);
			testInst.setDataset(testSet);
			
			for(int i=0;i<parts.length;i++){
				testInst.setValue(i, Double.parseDouble(parts[i]));
			}
			
			return testInst;			
			
		}
		
		
		

		@Override
		public void reduce(Text key, Iterator<Text> values,
				OutputCollector<Text, Text> output, Reporter rep)
				throws IOException {
			
			//�������ݣ�<keycuid��(x,y,time,wifi),(x,y,time,wifi),....>
			
			userPoiList.clear();
			statRes.clear();
			userPoiAndCluIDList.clear();
			
			while(values.hasNext()){
				String outstrstr = values.next().toString();
				userPoiList.add(outstrstr);
				
				//output.collect(key, new Text(outstrstr));
			}
			
			//ÿ���û���ÿ��poi���¼�������ϴغţ��γɸ�ʽ��(x,y,time,wifi,clusterID)
			//��������ڣ�userPoiAndCluIDList ��
			
			processUserInfo(userPoiList);			
			
			//���մغŽ��з��飺<cluseterID,(x,y,time,wifi,clusterID),(x,y,time,wifi,clusterID)>
			FeatureStat.init(userPoiAndCluIDList);	
			
			//ͳ��ÿ���ص����ĵ�
			FeatureStat.statCenterCordinate(statRes);
			
			//ͳ��ÿ���ص�wifi��			
			FeatureStat.statWifiEntropy(statRes);
			
			//ͳ��ÿ���ض�λ���ֵ����ӣ�ȥ�أ����Ϲ�ȥ�������ܶ�λ�����ӣ�ȥ�أ�
			
			FeatureStat.statRatioPresentDay(statRes);
			
			//ͳ��ÿ���أ����û���ƽ��ÿ��ͣ��ʱ��
			
			FeatureStat.statAvgStayTime(statRes);				

			//ͳ��ÿ���أ����û��İ���ռ��
			
			FeatureStat.statRatioDayTime(statRes);
			
			//ͳ��ÿ���أ����û�������ռ��
			
			FeatureStat.statRatioNightTime(statRes);				
			
			//ͳ��ÿ���أ����û�����ĩռ��
			
			FeatureStat.statRatioWeekend(statRes);
			
			
			//ͳ��ÿ�����е㣬�ֲ���8��ʱ�����ռ�����
			
			FeatureStat.statRatioTimeSlot(statRes,8);	
			
			/*
			 * ��ֹΪ�ˣ��Ѿ��γ��˴��û�cuid��������������Ϣ�ͱ�ǩ��Ϣ:statRes
			 *    (clusterID, <Centerx,centery,wifi_entropy,ratio_present_day,avg_stay_time,
			 *    ratio_daytime,ratio_night,ratio_weekend,ratio_slot_0,ratio_slot_1,ratio_slot_2,
			 *    ratio_slot_3,ratio_slot_4,ratio_slot_5,ratio_slot_6,ratio_slot_7,>)
			 */
			
			
			Set<Entry<String, ArrayList<String>>> enties = statRes.entrySet();
			
			ArrayList<String> userTmp=null;
			String clusterIDTmp=null;
			for(Entry<String, ArrayList<String>> en:enties){
				clusterIDTmp = en.getKey();
				userTmp = en.getValue();
				String feature = null;
				for(int i =0 ;i<userTmp.size();i++){					
					//�γɸ�ʽ��clusterID		xx	yy  entropy
					if(i==0){
						feature = userTmp.get(i)+"\t";
					}else if(i==(userTmp.size()-1)){
						feature +=userTmp.get(i);
					}else{
						feature += userTmp.get(i)+"\t";
					}
					
					//System.out.println(clusterIDTmp+"\t"+userTmp.get(i));
				}
				
				//System.out.println(clusterIDTmp+"\t"+feature);	
				
				output.collect(key, new Text(clusterIDTmp+"\t"+feature));
			}			

		}//end reduce
		
	}//end RedUserStat


	
	

}
