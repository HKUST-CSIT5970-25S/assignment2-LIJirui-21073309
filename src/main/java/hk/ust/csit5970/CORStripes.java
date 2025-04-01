package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

/**
 * Compute the bigram count using "pairs" approach
 */
public class CORStripes extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(CORStripes.class);

	/*
	 * TODO: write your first-pass Mapper here.
	 */
	private static class CORMapper1 extends
			Mapper<LongWritable, Text, Text, IntWritable> {
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			HashMap<String, Integer> wordCount = new HashMap<String, Integer>();
			// Please use this tokenizer! DO NOT implement a tokenizer by yourself!
			// 使用指定的分词器，清理文档
			String clean_doc = value.toString().replaceAll("[^a-z A-Z]", " ");
			StringTokenizer doc_tokenizer = new StringTokenizer(clean_doc);

			// 统计每个单词的出现次数
			while (doc_tokenizer.hasMoreTokens()) {
                String token = doc_tokenizer.nextToken().toLowerCase();
				if (!token.isEmpty()) {
					// 更新计数
					if (wordCount.containsKey(token)) {
						wordCount.put(token, wordCount.get(token) + 1);
					} else {
						wordCount.put(token, 1);
					}
				}
            }

			// 输出每个单词及其计数
			for (Iterator<Map.Entry<String, Integer>> it = wordCount.entrySet().iterator(); it.hasNext();) {
				Map.Entry<String, Integer> entry = it.next();
				context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
			}
		}
	}

	/*
	 * TODO: Write your first-pass reducer here.
	 */
	private static class CORReducer1 extends
			Reducer<Text, IntWritable, Text, IntWritable> {
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			// 计算总和
			int sum = 0;
			for (Iterator<IntWritable> it = values.iterator(); it.hasNext();) {
				sum += it.next().get();
			}
			// 输出结果
            context.write(key, new IntWritable(sum));
		}
	}

	/*
	 * TODO: Write your second-pass Mapper here.
	 */
	public static class CORStripesMapper2 extends Mapper<LongWritable, Text, Text, MapWritable> {
		@Override
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			Set<String> sorted_word_set = new TreeSet<String>();
			// Please use this tokenizer! DO NOT implement a tokenizer by yourself!
			String doc_clean = value.toString().replaceAll("[^a-z A-Z]", " ");
			StringTokenizer doc_tokenizers = new StringTokenizer(doc_clean);
			// 收集唯一单词
			while (doc_tokenizers.hasMoreTokens()) {
				sorted_word_set.add(doc_tokenizers.nextToken());
			}

			List<String> wordsList = new ArrayList<String>(sorted_word_set);
            int n = wordsList.size();
			// 对于每个单词 A，发出一个包含 A 后每个单词 B 的计数的条纹
            for (int i = 0; i < n; i++) {
                String wordA = wordsList.get(i);
                MapWritable stripe = new MapWritable();
                for (int j = i + 1; j < n; j++) {
                    String wordB = wordsList.get(j);
                    Text bText = new Text(wordB);
					// 初始化或增加 wordB 的计数
                    if (stripe.containsKey(bText)) {
                        IntWritable count = (IntWritable) stripe.get(bText);
                        count.set(count.get() + 1);
                    } else {
                        stripe.put(bText, new IntWritable(1));
                    }
                }
				// 如果条纹不为空，则输出
                if (!stripe.isEmpty()) {
                    context.write(new Text(wordA), stripe);
                }
            }
		}
	}

	/*
	 * TODO: Write your second-pass Combiner here.
	 */
	public static class CORStripesCombiner2 extends Reducer<Text, MapWritable, Text, MapWritable> {
		static IntWritable ZERO = new IntWritable(0);

		@Override
		protected void reduce(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
			/*
			 * TODO: Your implementation goes here.
			 */
			MapWritable combinedStripe = new MapWritable();
            for (MapWritable stripe : values) {
                for (Writable entryKey : stripe.keySet()) {
                    IntWritable count = (IntWritable) stripe.get(entryKey);
                    if (combinedStripe.containsKey(entryKey)) {
                        IntWritable existing = (IntWritable) combinedStripe.get(entryKey);
                        existing.set(existing.get() + count.get());
                    } else {
                        combinedStripe.put(entryKey, new IntWritable(count.get()));
                    }
                }
            }
            context.write(key, combinedStripe);
		}
	}

	/*
	 * TODO: Write your second-pass Reducer here.
	 */
	public static class CORStripesReducer2 extends Reducer<Text, MapWritable, PairOfStrings, DoubleWritable> {
		private static Map<String, Integer> word_total_map = new HashMap<String, Integer>();
		private static IntWritable ZERO = new IntWritable(0);

		/*
		 * Preload the middle result file.
		 * In the middle result file, each line contains a word and its frequency Freq(A), seperated by "\t"
		 */
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			Path middle_result_path = new Path("mid/part-r-00000");
			Configuration middle_conf = new Configuration();

			FileSystem fs = FileSystem.get(URI.create(middle_result_path.toString()), middle_conf);

			if (!fs.exists(middle_result_path)) {
				throw new IOException(middle_result_path.toString() + "文件不存在t!");
			}

			FSDataInputStream in = fs.open(middle_result_path);
			InputStreamReader inStream = new InputStreamReader(in);
			BufferedReader reader = new BufferedReader(inStream);

			LOG.info("开始读取...");
			String line;
			while ((line = reader.readLine()) != null) {
				String[] line_terms = line.split("\t");
				word_total_map.put(line_terms[0], Integer.valueOf(line_terms[1]));
				LOG.info("读取一行!");
			}
			reader.close();
			LOG.info("完成");
		}

		/*
		 * TODO: Write your second-pass Reducer here.
		 */
		@Override
		protected void reduce(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
			/*
			 * TODO: Your implementation goes here.
			 */
			// 聚合所有与键（词 A）相关的条带
            MapWritable combinedStripe = new MapWritable();
            for (MapWritable stripe : values) {
                for (Writable mapKey : stripe.keySet()) {
                    IntWritable count = (IntWritable) stripe.get(mapKey);
                    if (combinedStripe.containsKey(mapKey)) {
                        IntWritable existing = (IntWritable) combinedStripe.get(mapKey);
                        existing.set(existing.get() + count.get());
                    } else {
                        combinedStripe.put(mapKey, new IntWritable(count.get()));
                    }
                }
            }
			// 对于聚合条带中的每个词 B，计算相关系数
            Integer freqAObj = word_total_map.get(key.toString());
			if (freqAObj == null) {
				return;
			}
			int freqA = freqAObj;

            for (Writable mapKey : combinedStripe.keySet()) {
                String wordB = mapKey.toString();
				// 确保只输出 A < B 的对
                if (key.toString().compareTo(wordB) < 0) {
                    int freqB = word_total_map.containsKey(wordB) ? word_total_map.get(wordB) : 0;
                    if (freqA != 0 && freqB != 0) {
                        int freqAB = ((IntWritable) combinedStripe.get(mapKey)).get();
                        double correlation = (double) freqAB / (freqA * freqB);
                        context.write(new PairOfStrings(key.toString(), wordB), new DoubleWritable(correlation));
                    }
                }
            }
		}
	}

	/**
	 * Creates an instance of this tool.
	 */
	public CORStripes() {
	}

	private static final String INPUT = "input";
	private static final String OUTPUT = "output";
	private static final String NUM_REDUCERS = "numReducers";

	/**
	 * Runs this tool.
	 */
	@SuppressWarnings({ "static-access" })
	public int run(String[] args) throws Exception {
		Options options = new Options();

		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("input path").create(INPUT));
		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("output path").create(OUTPUT));
		options.addOption(OptionBuilder.withArgName("num").hasArg()
				.withDescription("number of reducers").create(NUM_REDUCERS));

		CommandLine cmdline;
		CommandLineParser parser = new GnuParser();

		try {
			cmdline = parser.parse(options, args);
		} catch (ParseException exp) {
			System.err.println("Error parsing command line: "
					+ exp.getMessage());
			return -1;
		}

		// Lack of arguments
		if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
			System.out.println("args: " + Arrays.toString(args));
			HelpFormatter formatter = new HelpFormatter();
			formatter.setWidth(120);
			formatter.printHelp(this.getClass().getName(), options);
			ToolRunner.printGenericCommandUsage(System.out);
			return -1;
		}

		String inputPath = cmdline.getOptionValue(INPUT);
		String middlePath = "mid";
		String outputPath = cmdline.getOptionValue(OUTPUT);

		int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
				.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

		LOG.info("Tool: " + CORStripes.class.getSimpleName());
		LOG.info(" - input path: " + inputPath);
		LOG.info(" - middle path: " + middlePath);
		LOG.info(" - output path: " + outputPath);
		LOG.info(" - number of reducers: " + reduceTasks);

		// Setup for the first-pass MapReduce
		Configuration conf1 = new Configuration();

		Job job1 = Job.getInstance(conf1, "Firstpass");

		job1.setJarByClass(CORStripes.class);
		job1.setMapperClass(CORMapper1.class);
		job1.setReducerClass(CORReducer1.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(IntWritable.class);

		FileInputFormat.setInputPaths(job1, new Path(inputPath));
		FileOutputFormat.setOutputPath(job1, new Path(middlePath));

		// Delete the output directory if it exists already.
		Path middleDir = new Path(middlePath);
		FileSystem.get(conf1).delete(middleDir, true);

		// Time the program
		long startTime = System.currentTimeMillis();
		job1.waitForCompletion(true);
		LOG.info("Job 1 Finished in " + (System.currentTimeMillis() - startTime)
				/ 1000.0 + " seconds");

		// Setup for the second-pass MapReduce

		// Delete the output directory if it exists already.
		Path outputDir = new Path(outputPath);
		FileSystem.get(conf1).delete(outputDir, true);


		Configuration conf2 = new Configuration();
		Job job2 = Job.getInstance(conf2, "Secondpass");

		job2.setJarByClass(CORStripes.class);
		job2.setMapperClass(CORStripesMapper2.class);
		job2.setCombinerClass(CORStripesCombiner2.class);
		job2.setReducerClass(CORStripesReducer2.class);

		job2.setOutputKeyClass(PairOfStrings.class);
		job2.setOutputValueClass(DoubleWritable.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(MapWritable.class);
		job2.setNumReduceTasks(reduceTasks);

		FileInputFormat.setInputPaths(job2, new Path(inputPath));
		FileOutputFormat.setOutputPath(job2, new Path(outputPath));

		// Time the program
		startTime = System.currentTimeMillis();
		job2.waitForCompletion(true);
		LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime)
				/ 1000.0 + " seconds");

		return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new CORStripes(), args);
	}
}