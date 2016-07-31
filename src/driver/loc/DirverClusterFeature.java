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
		
		job.setMapOutputKeyClass(Text.class);//设定map输出的key使用文本类，这样格式
		job.setMapOutputValueClass(Text.class);
		
		job.setInputFormat(TextInputFormat.class);//一行行的输入数据		
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
			fs.delete(outPath,true); //输出文件存在的情况下，删除此文件夹
		}
		
		FileOutputFormat.setOutputPath(job, outPath);
		
		//解析第三方包的参数		
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
		 * configure 完成mapper的初始化工作(non-Javadoc)
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
			output.collect(new Text(cuidStr),new Text(xx+"\t"+yy+"\t"+createTime+"\t"+wifi));//输出：<cuid,(x,y,time,wifi),()>	
  
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
		 * configure 完成reducer的初始化工作(non-Javadoc)
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
				
				//文件格式：<cuid,uid,homex,homey,comx,comy>
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
		 * 每个用户，聚类，打上簇号，打上标签（home/com）,输出
		 */
		public void processUserInfo(ArrayList<String> userPoiList,String cuid)throws IOException {
			
			if(userPoiList.size()<=0){
				
				System.out.println("userPoiList empty");
				return;
			}
			
			FastVector atts = new FastVector(2);
			atts.addElement(new Attribute("X"));//numberic类型
			atts.addElement(new Attribute("Y"));			
			Instances userPoiInstances = new Instances("userloc",atts,0);	
			String[] parts;
			
			//构造聚类使用的instances位置点数据
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
				inst.setValue(0,Double.parseDouble(logitude));//numberic 类型
				inst.setValue(1,Double.parseDouble(latitude));					
				inst.setDataset(userPoiInstances);
				userPoiInstances.add(inst);					
			}	
			
			//车主家和公司的真实位置:<homex homey comx comy>
			
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
			
			//聚类此用户的静态轨迹点数据
			DBSCAN clusterer = new DBSCAN(); 
			clusterer.setEpsilon(Double.parseDouble(epsion));
			clusterer.setMinPoints(Integer.parseInt(minPoint));	
			try{
				clusterer.buildClusterer(userPoiInstances);					
			}catch(Exception e){
				e.printStackTrace();
			}				
			
			// 评估
			ClusterEvaluation eval = new ClusterEvaluation();
			eval.setClusterer(clusterer);
			try{
				eval.evaluateClusterer(userPoiInstances);
			}catch(Exception e){
				e.printStackTrace();
			}
			
			//获取聚类集合中每一个实例，所属的簇号
			double []clusterAssign = eval.getClusterAssignments();
			int len = clusterAssign.length;
			
			double homeCluster = clusterAssign[len-2];
			double comCluster = clusterAssign[len-1];
			
			
			System.out.println("the clusterNUm:"+eval.getNumClusters());
			
			//每个记录一个实例，每个实例有一个簇号
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
			
			//接收数据：<keycuid，(x,y,time,wifi),(x,y,time,wifi),....>
			
			
			String cuidStr = key.toString();
			if(!driverRealLoc.containsKey(cuidStr)){
				//System.out.println(cuidStr);//没有车主的真实位置信息
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
			
			//每个用户的每条poi点记录，都加上簇号,同时打上家或公司的标签		
			
			processUserInfo(userPoiList,cuidStr);			
			
			//对于每个用户统计不同的特征指标 :userPoiList
			//<x,y,time,wifi,clusterID,label>
			FeatureStat.init(userPoiAndCluIDList);
			
			//<cluseterID,(x,y,time,wifi,clusterID),(x,y,time,wifi,clusterID)>
			
			
			//统计每个簇的中心点
			FeatureStat.statCenterCordinate(statRes);
			
			//统计每个簇的wifi熵			
			FeatureStat.statWifiEntropy(statRes);
			
			//统计每个簇定位出现的日子（去重）比上过去三个月总定位的日子（去重）
			
			FeatureStat.statRatioPresentDay(statRes);
			
			//统计每个簇，该用户的平均每天停留时长
			
			FeatureStat.statAvgStayTime(statRes);				

			//统计每个簇，该用户的白天占比
			
			FeatureStat.statRatioDayTime(statRes);
			
			//统计每个簇，该用户的晚上占比
			
			FeatureStat.statRatioNightTime(statRes);				
			
			//统计每个簇，该用户的周末占比
			
			FeatureStat.statRatioWeekend(statRes);
			
			
			//统计每个簇中点，分布在8个时间槽上占比情况
			
			FeatureStat.statRatioTimeSlot(statRes,8);	
			
			//给每条记录，打上家或公司的标签
			
			FeatureStat.addLabel(statRes);
			
			/*
			 * 截止为此，已经形成了此用户cuid：完整的特征信息和标签信息:statRes
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
					//形成格式：clusterID	 xx	yy entropy					
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
