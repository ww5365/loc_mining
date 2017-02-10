package driver.loc;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;




public class FeatureStat {
	
	private static HashMap<String, ArrayList<String>> userPoiGroupByCluster = new HashMap<String, ArrayList<String>>();
	

	/*
	 * @return 某个用户，按照簇号这个维度，进行分组数
	 * 
	 * 训练模型集合：
	 * <clusterID1,(x,y,time,Wifi,clusterID,label?),(x,y,time,Wifi,clusterID,label?),...>
	 * 
	 * 预测集合：
	 * <clusterID1,(x,y,time,Wifi,clusterID),(x,y,time,Wifi,clusterID),...>
	 */
	public static int init(ArrayList<String> userPoiList){
		
		//清空上一个用户的数据
		userPoiGroupByCluster.clear();
		
		//按照簇号，对此用户的数据进行分组
		
		if(userPoiList.size()<=0){
			return 0;
		}
		
		int clusterNum=0;
		String word;
		String[] parts;		
		
		for(int i=0;i<userPoiList.size();i++){
			word = userPoiList.get(i).trim();
			parts = word.split("\t");			
			String clusterID = parts[4].trim();			
			if(!userPoiGroupByCluster.containsKey(clusterID)){
				userPoiGroupByCluster.put(clusterID, new ArrayList<String>());	
				clusterNum++;
			}
			
			userPoiGroupByCluster.get(clusterID).add(word);	
		}
		
		return clusterNum;
	}
	
	
	/*
	 * @desc 统计用户的每个簇的中心点
	 */
	
	public static void statCenterCordinate(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userTmp = null;
		String clusterIDTmp = null;
		String[] parts;		
		
		for (Entry<String, ArrayList<String>> en : enties) {
			
			double maxLogitude = 0;
			double minLogitude = Double.MAX_VALUE;		
			double maxLatitude = 0;		
			double minLatitude = Double.MAX_VALUE;
			double x;
			double y;
			
			clusterIDTmp = en.getKey();
			userTmp = en.getValue();
			for (int i = 0; i < userTmp.size(); i++) {
				
				//System.out.println(clusterIDTmp + "\t" + userTmp.get(i));
				parts = userTmp.get(i).split("\t");
				
				x = Double.parseDouble(parts[0]);//x 坐标
				y = Double.parseDouble(parts[1]); //y 坐标
				
				if(x>maxLogitude){
					maxLogitude = x;
				}
				
				if(x<minLogitude){
					minLogitude = x;
				}
				
				if(y>maxLatitude){
					maxLatitude = y;
				}
				
				if(y<minLatitude){
					minLatitude = y;
				}				
			}
			
			BigDecimal xx = new BigDecimal((maxLogitude+minLogitude)/2);
			BigDecimal yy = new BigDecimal((maxLatitude+minLatitude)/2);
			
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			//将此用户所在簇的中心点，加入到特征list中
			statRes.get(clusterIDTmp).add(xx.toString());
			statRes.get(clusterIDTmp).add(yy.toString());			
			
		}		
		
		
	}
	
	

	/*
	 * @desc  统计用户每个簇的wifi熵数据
	 * 
	 * 计算方法：
	 * 
	 * 每个簇中出现的wifi数量，不去重，设为y；
	 * 每个簇中每个wifi的出现次数设为<x1,x2,x3...>；
	 * 
	 * 得到每个簇wifi的概率分布：<x1/y,x2/y,....>
	 * 
	 * 使用上面的分布，计算该簇的熵；
	 * 
	 */
	
	public static void statWifiEntropy(HashMap<String, ArrayList<String>> statRes){		
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	
		HashMap<String, Long> eachWifiCount = new HashMap<String, Long>();		
		ArrayList<Double> wifiRatio = new ArrayList<Double>();
		
		for (Entry<String, ArrayList<String>> en : enties) {
			
			wifiRatio.clear();
			eachWifiCount.clear();
			long wifiTotal = 0;			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();
			
			for (int i = 0; i < userPoiTmp.size(); i++) {
				//簇号维度的所有wifi数据统计
				parts = userPoiTmp.get(i).split("\t");
				wifiTotal++;
				String tmpWifi = parts[3].trim();
				if(!eachWifiCount.containsKey(tmpWifi)){
					eachWifiCount.put(tmpWifi, new Long(1));
				}else{
					Long tmpCount = eachWifiCount.get(tmpWifi);
					eachWifiCount.put(tmpWifi, ++tmpCount);
				}
			}
			
			//相同簇中，不同的wifi值所占的概率
			if(wifiTotal>0){
				Set<Entry<String, Long>> tmpEnties = eachWifiCount.entrySet();				
				for(Entry<String, Long> entmp:tmpEnties){
					Double ratio = new Double(entmp.getValue().doubleValue()/wifiTotal);
					wifiRatio.add(ratio);					
				}				
			}
			
			//System.out.println(clusterIDTmp+"\t"+wifiRatio);
			
			double entropy = UtilForMe.CalEntropy(wifiRatio);
			
			//System.out.println(clusterIDTmp+"\t"+entropy);
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			//将此用户所在簇的中心点，加入到特征list中
			statRes.get(clusterIDTmp).add(Double.toString(entropy));
		}		
	}
	
	

	

	/*
	 * @统计每个用户，在不同的簇中出现的日子 比上 过去三个月总共定位的日子
	 * 计算方法：
	 * 该用户，所有簇中，不重复的出现的日子，设为x；
	 * 某个簇中，不重复的出现的日子，设为y；
	 * 
	 * 结果： 每个簇：y/x
	 * 
	 */
	public static void statRatioPresentDay(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	

		HashSet<String> allDate = new HashSet<String>();
		HashMap<String,HashSet<String>> oneClusterPresentDay = new HashMap<String, HashSet<String>>();		
		oneClusterPresentDay.clear();
		allDate.clear();
		
		for (Entry<String, ArrayList<String>> en : enties) {			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();			
			
			for (int i = 0; i < userPoiTmp.size(); i++) {
				//簇号维度
				parts = userPoiTmp.get(i).split("\t");

				String tmpTimeStamp = parts[2].trim();
				
				String tmpDate = UtilForMe.timeStampToDate(tmpTimeStamp, "yyyy-MM-dd");	
				
				allDate.add(tmpDate);//记录所有簇中的不重复的时间
				
				if(!oneClusterPresentDay.containsKey(clusterIDTmp)){
					oneClusterPresentDay.put(clusterIDTmp, new HashSet<String>());
				}				
				oneClusterPresentDay.get(clusterIDTmp).add(tmpDate);//记录每个簇中，不重复的时间				
			}			
		}
		
		
		int allPresentDay = allDate.size();
		
		
		for (Entry<String, ArrayList<String>> en2 : enties) {			
			clusterIDTmp = en2.getKey();			
			int oneCluPresentDay = oneClusterPresentDay.get(clusterIDTmp).size();
			
			double ratioPresentDay = 0;
			
			if(allPresentDay!=0){
				ratioPresentDay = (double)oneCluPresentDay/allPresentDay;
			}
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			//将此用户所在簇的中心点，加入到特征list中
			statRes.get(clusterIDTmp).add(Double.toString(ratioPresentDay));
			
			//System.out.println("clusterIDTmp:ratio: "+ clusterIDTmp +"\t"+ratioPresentDay);
			
		}		
		
	}
	
	/*
	 * @统计每个用户，在不同的簇 ,  每天的平均停留时长
	 * 
	 * 计算方法：
	 * 特定簇中，出现的天数（去重） 设为y；
	 * 出现的这y天中，每天停留时长,计算总和设为x；
	 * 
	 * 结果：x/y
	 * 
	 */
	public static void statAvgStayTime(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	

		HashMap<String,Long> oneClusterMinStayTime = new HashMap<String, Long>();	
		HashMap<String,Long> oneClusterMaxStayTime = new HashMap<String, Long>();	

		
		for (Entry<String, ArrayList<String>> en : enties) {	
			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();	
			
			oneClusterMinStayTime.clear();
			oneClusterMaxStayTime.clear();
			
			for (int i = 0; i < userPoiTmp.size(); i++) {
				
				parts = userPoiTmp.get(i).split("\t");

				String tmpTimeStamp = parts[2].trim();
				
				String tmpDate = UtilForMe.timeStampToDate(tmpTimeStamp, "yyyy-MM-dd");	
				
				if(!oneClusterMinStayTime.containsKey(tmpDate)){
					oneClusterMinStayTime.put(tmpDate, new Long(Long.MAX_VALUE));
				}
				
				if(!oneClusterMaxStayTime.containsKey(tmpDate)){
					oneClusterMaxStayTime.put(tmpDate, new Long(0));
				}
				
				Long tmpTimeStampLong = new Long(tmpTimeStamp); 
				
				//寻找某天的最大时间和最小时间
				if(tmpTimeStampLong<oneClusterMinStayTime.get(tmpDate)){
					oneClusterMinStayTime.put(tmpDate, tmpTimeStampLong);
				}
				
				if(tmpTimeStampLong>oneClusterMaxStayTime.get(tmpDate)){
					oneClusterMaxStayTime.put(tmpDate, tmpTimeStampLong);
				}					
			}
			
			long totalTime = 0;
			long totalDay = 0;
			Set<Entry<String, Long>> minEntries = oneClusterMinStayTime.entrySet();
			
			for(Entry<String, Long> minEn:minEntries){
				String tmpDateKey = minEn.getKey();
				Long minTime = minEn.getValue();				
				Long maxTimeLong = oneClusterMaxStayTime.get(tmpDateKey);
				
				totalDay++;
				if(maxTimeLong-minTime>=0){
					totalTime+=(maxTimeLong.longValue()-minTime.longValue());
				}				
			}
			
			long avgStayTime = 0;  //每天平均停留时长 单位s
			if(totalDay!=0){
				avgStayTime = totalTime/totalDay;
			}
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			//将此用户所在簇的中心点，加入到特征list中
			statRes.get(clusterIDTmp).add(Long.toString(avgStayTime));
			
			//System.out.println("clusterIDTmp:avgStayTime:" + clusterIDTmp + "\t" + avgStayTime);			
			
		}		
	}
	
	/*
	 * @统计每个用户，不同簇 白天的日子比上 用户全部白天定位的总天数
	 * 
	 * 计算方法：
	 * 用户白天定位的天数（去重） 设为y；
	 * 某个簇中，白天的天数设为x；
	 * 
	 * 白天：6:00~18:00
	 * 
	 * 结果：x/y
	 * 
	 */
	public static void statRatioDayTime(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	

		String dayTimeBegin ="06:00:00";
		String dayTimeEnd ="18:00:00";
		
		long totalDayTime =0;

		HashMap<String,Long> oneClusterPresentDay = new HashMap<String, Long>();			
		oneClusterPresentDay.clear();

		
		for (Entry<String, ArrayList<String>> en : enties) {			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();			
			
			for (int i = 0; i < userPoiTmp.size(); i++) {
				//簇号维度的所有wifi数据统计
				parts = userPoiTmp.get(i).split("\t");

				String tmpTimeStamp = parts[2].trim();
				
				String tmpTime = UtilForMe.timeStampToDate(tmpTimeStamp, "HH:mm:ss");		
				
				//System.out.println("culstID:time:"+clusterIDTmp+"\t"+tmpTime);
				if(dayTimeBegin.compareToIgnoreCase(tmpTime)<=0&&dayTimeEnd.compareToIgnoreCase(tmpTime)>=0){
					totalDayTime++;					
					if(!oneClusterPresentDay.containsKey(clusterIDTmp)){
						oneClusterPresentDay.put(clusterIDTmp, new Long(0));
					}
					Long tmpLong = oneClusterPresentDay.get(clusterIDTmp);
					oneClusterPresentDay.put(clusterIDTmp, ++tmpLong); //记录每个簇中白天定位时间					
				}
			}//end for
		}//end for

		for (Entry<String, ArrayList<String>> en2 : enties) {			
			clusterIDTmp = en2.getKey();			
			Long tmpDayLong = oneClusterPresentDay.get(clusterIDTmp);
			
			if(tmpDayLong == null){
				tmpDayLong = new Long(0);
			}
			
			double ratioDayTime = 0;
			if(totalDayTime!=0){
				ratioDayTime =  (tmpDayLong.doubleValue())/totalDayTime;
			}
			
			//System.out.println("the cluster:ratioDayTime "+ clusterIDTmp +"\t"+ratioDayTime);	
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			//将此用户所在簇的中心点，加入到特征list中
			statRes.get(clusterIDTmp).add(Double.toString(ratioDayTime));
			
		}	
		
	}
	
	
	
	/*
	 * @统计每个用户，不同簇 晚上的日子比上 用户全部晚上定位的总天数
	 * 
	 * 计算方法：
	 * 用户晚上定位的天数（去重） 设为y；
	 * 某个簇中，晚上的天数设为x；
	 * 
	 * 晚上：0：00~6：00  18:00~23:59
	 * 
	 * 结果：x/y
	 * 
	 */
	public static void statRatioNightTime(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	

		String dayTimeBegin ="06:00:00";
		String dayTimeEnd ="18:00:00";
		
		long totalNightTime =0;

		HashMap<String,Long> oneClusterPresentDay = new HashMap<String, Long>();			
		oneClusterPresentDay.clear();

		
		for (Entry<String, ArrayList<String>> en : enties) {			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();			
			
			for (int i = 0; i < userPoiTmp.size(); i++) {
				//簇号维度的所有wifi数据统计
				parts = userPoiTmp.get(i).split("\t");

				String tmpTimeStamp = parts[2].trim();
				
				String tmpTime = UtilForMe.timeStampToDate(tmpTimeStamp, "HH:mm:ss");		
				
				//System.out.println("culstID:time:"+clusterIDTmp+"\t"+tmpTime);
				if(!(dayTimeBegin.compareToIgnoreCase(tmpTime)<=0&&dayTimeEnd.compareToIgnoreCase(tmpTime)>=0)){
					totalNightTime++;					
					if(!oneClusterPresentDay.containsKey(clusterIDTmp)){
						oneClusterPresentDay.put(clusterIDTmp, new Long(0));
					}
					Long tmpLong = oneClusterPresentDay.get(clusterIDTmp);
					oneClusterPresentDay.put(clusterIDTmp, ++tmpLong); //记录每个簇中白天定位时间					
				}
			}//end for
		}//end for

		for (Entry<String, ArrayList<String>> en2 : enties) {			
			clusterIDTmp = en2.getKey();			
			Long tmpNightLong = oneClusterPresentDay.get(clusterIDTmp);
			if(tmpNightLong == null){
				tmpNightLong = new Long(0);
			}
			
			double ratioNightTime = 0;
			if(totalNightTime!=0){
				ratioNightTime =  (tmpNightLong.doubleValue())/totalNightTime;
			}
			
			//System.out.println("the cluster:ratioDayTime "+ clusterIDTmp +"\t"+ratioNightTime);	
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			statRes.get(clusterIDTmp).add(Double.toString(ratioNightTime));
			
		}			
	}
	
	
	
	/*
	 * @统计每个用户，不同簇 周末的日子比上 用户全部周末定位的总天数
	 * 
	 * 计算方法：
	 * 用户周末定位的天数（去重） 设为y；
	 * 某个簇中，周末的天数设为x；
	 * 
	 * 晚上：0：00~6：00  18:00~23:59
	 * 
	 * 结果：x/y
	 * 
	 */
	public static void statRatioWeekend(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;			
		long totalWeekend =0;
		HashMap<String,Long> oneClusterPresentDay = new HashMap<String, Long>();			
		oneClusterPresentDay.clear();
		
		for (Entry<String, ArrayList<String>> en : enties) {			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();			
			
			for (int i = 0; i < userPoiTmp.size(); i++) {
				//簇号维度的所有wifi数据统计
				parts = userPoiTmp.get(i).split("\t");

				String tmpTimeStamp = parts[2].trim();
				
				boolean isWeekend = UtilForMe.timeStampIsWeekend(tmpTimeStamp);		
				
				//System.out.println("culstID:time:"+clusterIDTmp+"\t"+tmpTime);
				if(isWeekend){
					totalWeekend++;					
					if(!oneClusterPresentDay.containsKey(clusterIDTmp)){
						oneClusterPresentDay.put(clusterIDTmp, new Long(0));
					}
					Long tmpLong = oneClusterPresentDay.get(clusterIDTmp);
					oneClusterPresentDay.put(clusterIDTmp, ++tmpLong); //记录每个簇中weekend定位时间					
				}
			}//end for
		}//end for

		for (Entry<String, ArrayList<String>> en2 : enties) {			
			clusterIDTmp = en2.getKey();			
			Long tmpWeekendLong = oneClusterPresentDay.get(clusterIDTmp);
			if(tmpWeekendLong == null){
				tmpWeekendLong = new Long(0);
			}
			
			double ratioWeekend = 0;
			if(totalWeekend!=0){
				ratioWeekend =  (tmpWeekendLong.doubleValue())/totalWeekend;
			}
			
			//System.out.println("the cluster:ratioWeekend "+ clusterIDTmp +"\t"+ratioWeekend);	
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			statRes.get(clusterIDTmp).add(Double.toString(ratioWeekend));
			
		}			
	}
	
	
	/*
	 * @desc 
	 * 将一天24h分成n个time slot;
	 * 计算每个time-slot中，用户cuid的每个簇中出现的总点数，假设为y
	 * 本簇中，在各个时间槽中出现的点<x1,x2,....>，占本簇所有点(y)的比例
	 * 
	 * @para  slot  应该可以被24整除
	 */
	
	

	public static void statRatioTimeSlot(HashMap<String, ArrayList<String>> statRes,int slot){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	


		long[] eachSlotCount = new long [slot];
		
		for (Entry<String, ArrayList<String>> en : enties) {			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();	
			
			//每个簇初始化一次
			for(int i=0;i<slot;i++){
				eachSlotCount[i] = 0;
			}
			
			long totalCount = userPoiTmp.size();//此簇中所有点的个数
			
			for(int i=0;i<userPoiTmp.size();i++){
				
				parts = userPoiTmp.get(i).split("\t");

				String tmpTimeStamp = parts[2].trim();
				
				String tmpHour = UtilForMe.timeStampToDate(tmpTimeStamp, "HH");	
				
				//System.out.println("tmpHour: "+tmpHour);
				
				int eachSlot = 24/slot;  //每个槽中，占用的小时个数
				
				int index = Integer.parseInt(tmpHour)/eachSlot;
				eachSlotCount[index]++;				
			}
			
			//计算每个时间槽出现的该簇中点，所占该簇所有点的占比
			for(int i=0;i<slot;i++){
				
				double ratioSlot = 0;
				if(totalCount!=0){
					ratioSlot = (double)eachSlotCount[i]/totalCount;
				}
				
				//System.out.println("eachslotcount: ratio "+eachSlotCount[i] +"\t" + ratioSlot);
				
				if(!statRes.containsKey(clusterIDTmp)){
					statRes.put(clusterIDTmp, new ArrayList<String>());
				}				
				statRes.get(clusterIDTmp).add(Double.toString(ratioSlot));			
			}			
		}
		
	}
	
	/*
	 * @desc 
	 * <clusterID1,(x,y,time,Wifi,clusterID,label?),(x,y,time,Wifi,clusterID,label?),...>
	 * 
	 * 给每个簇，打上家或公司的标签
	 * 
	 */
	
	public static void addLabel(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	
		
		for (Entry<String, ArrayList<String>> en : enties) {			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();	
			
			parts = userPoiTmp.get(0).split("\t");
			
			String labelStr = null;
			
			//System.out.println("parts len: "+parts.length);
			
			if(parts.length==6){
				labelStr = parts[5];
			}else{
				labelStr = "";
			}		
			
			// System.out.println("eachslotcount: ratio "+eachSlotCount[i] +"\t"
			// + ratioSlot);
			
			//System.out.println(clusterIDTmp+"\t"+userPoiTmp);

			if (!statRes.containsKey(clusterIDTmp)) {
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			statRes.get(clusterIDTmp).add(labelStr);				
		}
		
	}
	
	
	
	
	
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		String s1 = "11796016.000000	3011751.000000	1440900000	93351051973200	0	H";  //8.30 10:00:00
		String s2 = "11796018.000000	3011753.000000	1440471600	93351051973210	0	H";  //8.25 11:00:00
		String s8 = "11796018.000000	3011753.000000	1440586800	93351051973210	0	H";  //8.26 19:00:00
		
		String s3 = "11796018.000000	3011751.000000	1440640800	93351051973200	1	O";  // 8.27 10:00
		String s4 = "11796019.000000	3011741.000000	1440642600	93351051973100	1	O";  // 8.27 10:30:00
		String s5 = "11796019.000000	3011741.000000	1440586800	93351051973300	1	O";  // 8.26 19:00:00
		String s6 = "11796019.000000	3011741.000000	1440781201	93351051973100	1	O";  // 8.29 1:00:01
		String s7 = "11796019.000000	3011741.000000	1440460800	93351051973300	1	O";  // 8.25 8:00:00
		
		
		ArrayList<String> poiuser = new ArrayList<String>();
		
		poiuser.add(s1);
		poiuser.add(s2);
		poiuser.add(s3);
		poiuser.add(s4);
		poiuser.add(s5);
		poiuser.add(s6);
		poiuser.add(s7);
		poiuser.add(s8);
		
		
		
		FeatureStat.init(poiuser);		
		
		HashMap<String, ArrayList<String>> statRes = new HashMap<String, ArrayList<String>>() ;
		
		FeatureStat.statCenterCordinate(statRes);		
		
		FeatureStat.statWifiEntropy(statRes);
		
		FeatureStat.statRatioPresentDay(statRes);
		
		FeatureStat.statAvgStayTime(statRes);
		
		FeatureStat.statRatioDayTime(statRes);
		
		FeatureStat.statRatioNightTime(statRes);
		
		FeatureStat.statRatioWeekend(statRes);
		
		FeatureStat.statRatioTimeSlot(statRes,8);
		
		
		FeatureStat.addLabel(statRes);
		
		
		
		
		Set<Entry<String, ArrayList<String>>> enties = statRes.entrySet();
		
		ArrayList<String> userTmp=null;
		String clusterIDTmp=null;
		for(Entry<String, ArrayList<String>> en:enties){
			clusterIDTmp = en.getKey();
			userTmp = en.getValue();
			String feature = null;
			for(int i =0 ;i<userTmp.size();i++){
				
				//形成格式：clusterID		xx	yy  entropy
				if(i==0){
					feature = userTmp.get(i)+"\t";
				}else if(i==(userTmp.size()-1)){
					feature +=userTmp.get(i);
				}else{
					feature += userTmp.get(i)+"\t";
				}
				
				//System.out.println(clusterIDTmp+"\t"+userTmp.get(i));
			}
			
			System.out.println(clusterIDTmp+"\t"+feature);	
		}
		

		
		
		
	}

}
