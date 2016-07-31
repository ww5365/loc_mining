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


public class UserClusterFeatureV2 {


	/**
	 * @param args
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		
		JobConf job = new JobConf();
		job.setJarByClass(UserClusterFeatureV2.class);
		
		job.setMapOutputKeyClass(Text.class);//设定map输出的key使用文本类，这样格式
		job.setMapOutputValueClass(Text.class);
		
		job.setInputFormat(TextInputFormat.class);//一行行的输入数据		
		job.setOutputFormat(TextOutputFormat.class);
		
		job.setMapperClass(MapUserCluster.class);
		job.setReducerClass(RedUserStat.class);
		
		job.set("mapred.job.priority", "VERY_HIGH");
		job.set("mapred.job.queue.name", "map-international");
		
		
		job.setNumReduceTasks(5);
		
		//设置hadoop平台上的输入输出文件
		FileInputFormat.setInputPaths(job,"/app/map/map-international/wangwei/tmp/staypoi_test");
		Path outPath = new Path("/app/map/map-international/wangwei/tmp/user_classify");
		FileSystem fs = outPath.getFileSystem(job);
		
		if(fs.exists(outPath)){
			fs.delete(outPath,true); //输出文件存在的情况下，删除此文件夹
		}
		
		FileOutputFormat.setOutputPath(job, outPath);
		
		//设置使用hadoop平台，临时文件
		
		job.set("mapred.create.symlink", "yes");
		job.set("mapred.cache.files", "/app/map/map-international/wangwei/tmp/model/j48.model#j48.model");

		
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
				
				epsion = "0.008";				
				minPoint = "10";
				
				//读取序列化数据：反序列化
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
		 * 每个用户，聚类，打上簇号，输出
		 */
		public void processUserInfo(ArrayList<String> userPoiList)throws IOException {
			
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
				String logitude = parts[0].trim();   //这是用户的经度
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
			
			//每个记录一个实例，每个实例有一个簇号
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
		 * 构造特征值的Instance
		 */
		
		public  Instance makeInstance(String feature,Instances testSet){
			
			if(feature==null){
				return null;
			}
			String[] parts = feature.split("\t");
			
			if(parts.length!=14){
				
				System.out.println("parts len: "+ parts.length);
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
			
			//接收数据：<keycuid，(x,y,time,wifi),(x,y,time,wifi),....>
			
			userPoiList.clear();
			statRes.clear();
			userPoiAndCluIDList.clear();
			
			while(values.hasNext()){
				String outstrstr = values.next().toString();
				userPoiList.add(outstrstr);
				
				//output.collect(key, new Text(outstrstr));
			}
			
			//每个用户的每条poi点记录，都加上簇号
			processUserInfo(userPoiList);			
			
			//对于每个用户统计不同的特征指标 :userPoiList
			//<x,y,time,wifi,clusterID>
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
			
			
			
			/*
			 * 截止为此，已经形成了此用户cuid：完整的特征信息和标签信息:statRes
			 *    (clusterID, <Centerx,centery,wifi_entropy,ratio_present_day,avg_stay_time,
			 *    ratio_daytime,ratio_night,ratio_weekend,ratio_slot_0,ratio_slot_1,ratio_slot_2,
			 *    ratio_slot_3,ratio_slot_4,ratio_slot_5,ratio_slot_6,ratio_slot_7,>)
			 */
			
			
			
			Set<Entry<String, ArrayList<String>>> enties = statRes.entrySet();
			
			ArrayList<String> userFeature=null;
			String clusterIDTmp = null;
			String centerX = null;
			String centerY = null;
			
			
			/*
			 * 读取序列化的：模型文件：j48
			 * 
			 * @relation driverModel   
			 * @attribute wifi_entropy numeric
			 * @attribute present_day numeric
			 * @attribute avg_stay_time numeric
			 * @attribute day_time numeric
			 * @attribute night_time numeric
			 * @attribute weekend numeric
			 * @attribute slot_time_0 numeric
			 * @attribute slot_time_1 numeric
			 * @attribute slot_time_2 numeric
			 * @attribute slot_time_3 numeric
			 * @attribute slot_time_4 numeric
			 * @attribute slot_time_5 numeric
			 * @attribute slot_time_6 numeric
			 * @attribute slot_time_7 numeric
			 * @attribute label {H,C,O}
			 */
			
			Object obj[]= null;			
			try {
				obj = SerializationHelper.readAll("j48.model");
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			Classifier tree = (Classifier)obj[0];
			Instances trainSet = (Instances) obj[1];
			
			Instances testSet = trainSet.stringFreeStructure();  //最后一列，不需要设置值，但属性要有	
			
			String homeRes = null;
			String comRes = null;
			String homeClusterIDRes = null;
			String comClusterIDRes = null;
			
			double lastHome = 0;
			double lastCom = 0;	
			
			System.out.println("cuid:"+key.toString());
			
			for(Entry<String, ArrayList<String>> en:enties){
				clusterIDTmp = en.getKey();
				userFeature = en.getValue();
				String feature = null;
				
				if("-1".equals(clusterIDTmp)){
					//簇号为-1，立群点；不分类
					continue;
				}
				
				for(int i =0 ;i<userFeature.size();i++){
					//形成格式：clusterID	 xx	yy entropy					
					if(i==0){
						centerX = userFeature.get(i);  //取特征结果中的中心点x					
					}else if(i==1){
						centerY = userFeature.get(i);  //取特征结果中的中心点y
					}else if(i==2){
						feature = userFeature.get(i)+"\t"; //取特征值			 			
					}else if(i==(userFeature.size()-1)){
						feature += userFeature.get(i);
					}else{
						feature += userFeature.get(i)+"\t";
					}
				}
				
				System.out.println(userFeature);
						
				Instance testInst = makeInstance(feature,testSet);
				
				System.out.println("testInst: " +testInst);
				
				if(!trainSet.equalHeaders(testSet)){
					System.out.println("train and test set header not compatible!");
					continue;
				}
				
				double pred = -1;
				double[] dist = null;
				String preClass = null;
				
				try {
					pred = tree.classifyInstance(testInst); // 将测试集中的实例预测什么类别，索引值
					dist = tree.distributionForInstance(testInst);
					preClass = testSet.classAttribute().value((int) pred);

					System.out.println("pre:preClass " + pred + " " + preClass);

					System.out.println(Utils.arrayToString(dist));

					// System.out.println("pred:dist: "+pred + "\t"+ preClass
					// +"\t"+ dist);

					if (pred == 0 && dist[(int) pred] > lastHome) {// 预测为家且置信度最大的记录
						homeRes = new String(centerX + "\t" + centerY + "\t" + preClass);
						lastHome = dist[(int) pred];
						
						homeClusterIDRes = clusterIDTmp;

					}

					if (pred == 1 && dist[(int) pred] > lastCom) {// 预测为公司且置信度最大的记录
						comRes = new String(centerX + "\t" + centerY + "\t" + preClass);
						lastCom = dist[(int) pred];
						
						comClusterIDRes = clusterIDTmp;
					}
					
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					continue;
				}
			}
			
			//每个用户只有一个家或公司的结果
			if (!("-1".equals(homeClusterIDRes)) && homeRes != null) {
				output.collect(key, new Text(homeClusterIDRes + "\t"+ homeRes));
			}

			if (!("-1".equals(comClusterIDRes)) && comRes != null) {
				output.collect(key, new Text(comClusterIDRes + "\t"+ comRes));
			}
			

		}//end reduce
		
	}//end RedUserStat


	
	

}
