package driver.loc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
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

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.DBSCAN;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


public class DirverClusterFeature {


	/**
	 * @param args
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		
		JobConf job = new JobConf();
		job.setJarByClass(DirverClusterFeature.class);
		
		job.setMapOutputKeyClass(Text.class);//�趨map�����keyʹ���ı��࣬������ʽ
		job.setMapOutputValueClass(Text.class);
		
		job.setInputFormat(TextInputFormat.class);//һ���е���������		
		job.setOutputFormat(TextOutputFormat.class);
		
		job.setMapperClass(MapUserCluster.class);
		job.setReducerClass(RedUserStat.class);
		
		job.set("mapred.job.priority", "VERY_HIGH");
		job.set("mapred.job.queue.name", "map-international");
		
		job.set("mapred.create.symlink", "yes");
		//job.set("mapred.cache.files", "/app/map/map-international/wangwei/tmp/driver_real_loc/driver_real_loc_test#driver_real_loc_test");
		job.set("mapred.cache.files", "/app/map/map-international/wangwei/tmp/driver_real_loc/driver_real_loc#driver_real_loc");
		
		//job.setNumReduceTasks(100);
		job.setNumReduceTasks(10);
		
		//FileInputFormat.setInputPaths(job,"/app/map/map-international/wangwei/tmp/driver_staypoi");
		FileInputFormat.setInputPaths(job,"/app/map/map-international/wangwei/tmp/driver_stay_test");
		
		Path outPath = new Path("/app/map/map-international/wangwei/tmp/driver_feature_test");
		FileSystem fs = outPath.getFileSystem(job);
		
		if(fs.exists(outPath)){
			fs.delete(outPath,true); //����ļ����ڵ�����£�ɾ�����ļ���
		}
		
		FileOutputFormat.setOutputPath(job, outPath);
		
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
			
			String cuidStr = parts[0].trim();
			String xx = parts[1].trim();
			String yy = parts[2].trim();
			String createTime = parts[3].trim();
			String wifi = parts[4].trim();				
			output.collect(new Text(cuidStr),new Text(xx+"\t"+yy+"\t"+createTime+"\t"+wifi));//�����<cuid,(x,y,time,wifi),()>	
  
		}	
		
	}
	
	
	/*
	 * @reducer
	 */
	
	static class RedUserStat extends MapReduceBase implements Reducer<Text, Text, Text, Text>{
		
		public ArrayList<String> userPoiList;
		public ArrayList<String> userPoiAndCluIDList;		
		public HashMap<String, ArrayList<String>> statRes;
		public HashMap<String, String> driverRealLoc;
		
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
				driverRealLoc = new HashMap<String, String>();
				
				epsion = "0.008";				
				minPoint = "10";
				
				Path path[]=null;
				
				path=DistributedCache.getLocalCacheFiles(context);
				
				//System.out.println("get file path"+path[0].toString());  
				
				//�ļ���ʽ��<cuid,uid,homex,homey,comx,comy>
				//FileReader reader = new FileReader(new File("driver_real_loc_test"));
				FileReader reader = new FileReader(new File("driver_real_loc"));
				
				BufferedReader br = new BufferedReader(reader);
		        String line = null;
		        String[] parts;
		        while ((line = br.readLine()) != null) {
		            //System.out.println("real loc:"+line);
		            
		            parts = line.split("\t");
		            String cuidStr = parts[0].trim();		            
		            String homeX = parts[2].trim();
		            String homeY = parts[3].trim();
		            String comX = parts[4].trim();
		            String comY = parts[5].trim();	            
		            
		            if(!driverRealLoc.containsKey(cuidStr)){
		            	driverRealLoc.put(cuidStr, new String());		            
		            }
		            
		            //String valueStr = new String();
		            
		            driverRealLoc.put(cuidStr,homeX +"\t"+homeY +"\t"+comX +"\t"+comY);           
		            
		        }
		        br.close();
		        reader.close();				
				
				
	    		System.out.println("reducer configure success!!!");
			}
			catch(Exception e){
				e.printStackTrace();
				System.err.println("reducer configure error!!");
			}
		}
		
		
		/*
		 * ÿ���û������࣬���ϴغţ����ϱ�ǩ��home/com��,���
		 */
		public void processUserInfo(ArrayList<String> userPoiList,String cuid)throws IOException {
			
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
				String logitude = parts[0].trim();
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
			
			//�����Һ͹�˾����ʵλ��:<homex homey comx comy>
			
			String driverLocStr = driverRealLoc.get(cuid);
			String[] cordinateStr = driverLocStr.split("\t");
			
			String homeX = cordinateStr[0].trim();
			String homeY = cordinateStr[1].trim();
			String comX = cordinateStr[2].trim();
			String comY = cordinateStr[3].trim();	
			
			Instance home = new Instance(2);
			Instance com = new Instance(2);
			
			home.setValue(0,Double.parseDouble(homeX));
			home.setValue(1,Double.parseDouble(homeY));
			
			com.setValue(0,Double.parseDouble(comX));
			com.setValue(1,Double.parseDouble(comY));
			
			home.setDataset(userPoiInstances);
			userPoiInstances.add(home);
			
			com.setDataset(userPoiInstances);
			userPoiInstances.add(com);
			
			System.out.println("cuid:home:com: "+cuid+"\t"+driverLocStr);
			
			System.out.println("Instances: "+userPoiInstances);
			
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
			int len = clusterAssign.length;
			
			double homeCluster = clusterAssign[len-2];
			double comCluster = clusterAssign[len-1];
			
			
			System.out.println("the clusterNUm:"+eval.getNumClusters());
			
			//ÿ����¼һ��ʵ����ÿ��ʵ����һ���غ�
			for(int i=0;i<userPoiList.size();i++){
				
				parts=userPoiList.get(i).split("\t");
				String logiTmp = parts[0].trim();
				String latiTmp = parts[1].trim();
				String timeTmp = parts[2].trim();
				String wifiTmp = parts[3].trim();
				String clusterTmp = Integer.toString((int)clusterAssign[i]);
				String label = new String();
				if(homeCluster == clusterAssign[i]){
					label="H";
				}else if(comCluster == clusterAssign[i]){
					label="C";
				}else{
					label="O";
				}
				//String clusterTmp = Integer.toString((int)0);
				//output.collect(new Text(cuidTmp), new Text(logiTmp+"\t"+latiTmp+"\t"+timeTmp+"\t"+wifiTmp+"\t"+clusterTmp));
				//System.out.println(logiTmp+"\t"+latiTmp+"\t"+timeTmp+"\t"+wifiTmp+"\t"+clusterTmp+"\t"+label);
				userPoiAndCluIDList.add(new String(logiTmp+"\t"+latiTmp+"\t"+timeTmp+"\t"+wifiTmp+"\t"+clusterTmp+"\t"+label));			
			}		
			
		}	
		
		

		@Override
		public void reduce(Text key, Iterator<Text> values,
				OutputCollector<Text, Text> output, Reporter rep)
				throws IOException {
			
			//�������ݣ�<keycuid��(x,y,time,wifi),(x,y,time,wifi),....>
			
			
			String cuidStr = key.toString();
			if(!driverRealLoc.containsKey(cuidStr)){
				//System.out.println(cuidStr);//û�г�������ʵλ����Ϣ
				return;
			}		
			
			userPoiList.clear();
			statRes.clear();
			userPoiAndCluIDList.clear();
			
			while(values.hasNext()){
				String outstrstr = values.next().toString();
				userPoiList.add(outstrstr);
				
				//output.collect(key, new Text(outstrstr));
			}
			
			//ÿ���û���ÿ��poi���¼�������ϴغ�,ͬʱ���ϼһ�˾�ı�ǩ		
			
			processUserInfo(userPoiList,cuidStr);			
			
			//����ÿ���û�ͳ�Ʋ�ͬ������ָ�� :userPoiList
			//<x,y,time,wifi,clusterID,label>
			FeatureStat.init(userPoiAndCluIDList);
			
			//<cluseterID,(x,y,time,wifi,clusterID),(x,y,time,wifi,clusterID)>
			
			
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
			
			//��ÿ����¼�����ϼһ�˾�ı�ǩ
			
			FeatureStat.addLabel(statRes);
			
			/*
			 * ��ֹΪ�ˣ��Ѿ��γ��˴��û�cuid��������������Ϣ�ͱ�ǩ��Ϣ:statRes
			 *    (clusterID, <Centerx,centery,wifi_entropy,ratio_present_day,avg_stay_time,
			 *    ratio_daytime,ratio_night,ratio_weekend,ratio_slot_0,ratio_slot_1,ratio_slot_2,
			 *    ratio_slot_3,ratio_slot_4,ratio_slot_5,ratio_slot_6,ratio_slot_7,label>)
			 */
			
			Set<Entry<String, ArrayList<String>>> enties = statRes.entrySet();
			
			ArrayList<String> userFeature=null;
			String clusterIDTmp=null;
			for(Entry<String, ArrayList<String>> en:enties){
				clusterIDTmp = en.getKey();
				userFeature = en.getValue();
				String feature = null;
				for(int i =0 ;i<userFeature.size();i++){
					//�γɸ�ʽ��clusterID	 xx	yy entropy					
					if(i==0){
						feature = userFeature.get(i)+"\t";
					}else if(i==(userFeature.size()-1)){
						feature +=userFeature.get(i);
					}else{
						feature += userFeature.get(i)+"\t";
					}
				}
				
				//System.out.println(clusterIDTmp+"\t"+feature);	
				
				if(!("-1".equals(clusterIDTmp))){
					output.collect(key, new Text(clusterIDTmp+"\t"+feature));
				}
				
			}

		}//end reduce
		
	}//end RedUserStat


	
	

}
