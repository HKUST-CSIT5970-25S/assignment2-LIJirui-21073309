package hk.ust.csit5970;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

/**
 * Compute the bigram count using the "stripes" approach
 */
public class BigramFrequencyStripes extends Configured implements Tool {
	private static final Logger LOG = Logger
			.getLogger(BigramFrequencyStripes.class);

	/*
	 * Mapper: emits <word, stripe> where stripe is a hash map
	 */
	private static class MyMapper extends
			Mapper<LongWritable, Text, Text, HashMapStringIntWritable> {

		// Reuse objects to save overhead of object creation.
		private static final Text KEY = new Text();
		private static final HashMapStringIntWritable STRIPE = new HashMapStringIntWritable();

		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String line = ((Text) value).toString();
			String[] words = line.trim().split("\\s+");

			/*
			 * TODO: Your implementation goes here.
			 */
			// 如果单词数量少于2，直接返回
			if(words.length < 2) 
				return; 
            for (int i = 0; i < words.length - 1; i++) {
                String currentWord = cleanWord(words[i]);
                String nextWord = cleanWord(words[i + 1]);

				// 如果当前单词或下一个单词为空，继续下一次循环
                if (currentWord.isEmpty() || nextWord.isEmpty()) {
                    continue;
                }

                KEY.set(currentWord);
                STRIPE.clear();
                STRIPE.put(nextWord, 1);
                context.write(KEY, STRIPE);
            }
		}

		// 清理单词，去除不必要的字符
		private String cleanWord(String word) {
			return word.replaceAll("[^a-zA-Z']", "")
					.replaceAll("^'+", "")
					.replaceAll("'+$", "");
		}
	}

	/*
	 * TODO: write your reducer to aggregate all stripes associated with each key
	 */
	private static class MyReducer extends
			Reducer<Text, HashMapStringIntWritable, PairOfStrings, FloatWritable> {

		// Reuse objects.
		private final static HashMapStringIntWritable SUM_STRIPES = new HashMapStringIntWritable();
		private final static PairOfStrings BIGRAM = new PairOfStrings();
		private final static FloatWritable FREQ = new FloatWritable();

		@Override
		public void reduce(Text key,
				Iterable<HashMapStringIntWritable> stripes, Context context)
				throws IOException, InterruptedException {
			/*
			 * TODO: Your implementation goes here.
			 */
			// 清空总条纹
			SUM_STRIPES.clear();

			// 汇总所有条纹
            for (HashMapStringIntWritable stripe : stripes) {
                for (Map.Entry<String, Integer> entry : stripe.entrySet()) {
                    String b = entry.getKey();
                    int count = entry.getValue();
                    SUM_STRIPES.increment(b, count);
                }
            }

            int sumTotal = 0;
			// 计算总计数
            for (Map.Entry<String, Integer> entry : SUM_STRIPES.entrySet()) {
                sumTotal += entry.getValue();
            }
            if (sumTotal == 0) {
                return;
            }

			// 输出当前单词的总计数
            BIGRAM.set(key.toString(), "");
            FREQ.set(sumTotal);
            context.write(BIGRAM, FREQ);

			// 输出每个二元组及其相对频率
            for (Map.Entry<String, Integer> entry : SUM_STRIPES.entrySet()) {
                String b = entry.getKey();
                int count = entry.getValue();
                float frequency = (float) count / sumTotal;

                BIGRAM.set(key.toString(), b);
                FREQ.set(frequency);
                context.write(BIGRAM, FREQ);
			}
		}
	}

	/*
	 * TODO: Write your combiner to aggregate all stripes with the same key
	 */
	private static class MyCombiner
			extends
			Reducer<Text, HashMapStringIntWritable, Text, HashMapStringIntWritable> {
		// Reuse objects.
		private final static HashMapStringIntWritable SUM_STRIPES = new HashMapStringIntWritable();

		@Override
		public void reduce(Text key,
				Iterable<HashMapStringIntWritable> stripes, Context context)
				throws IOException, InterruptedException {
			/*
			 * TODO: Your implementation goes here.
			 */
			// 清空总条纹
			SUM_STRIPES.clear();

			// 汇总所有条纹
            for (HashMapStringIntWritable stripe : stripes) {
                for (Map.Entry<String, Integer> entry : stripe.entrySet()) {
                    String b = entry.getKey();
                    int count = entry.getValue();
                    SUM_STRIPES.increment(b, count);
                }
            }

			// 输出合并后的条纹
            HashMapStringIntWritable outStripe = new HashMapStringIntWritable();
			outStripe.putAll(SUM_STRIPES);
			context.write(key, outStripe);
		}
	}

	/**
	 * Creates an instance of this tool.
	 */
	public BigramFrequencyStripes() {
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
		String outputPath = cmdline.getOptionValue(OUTPUT);
		int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
				.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

		LOG.info("Tool: " + BigramFrequencyStripes.class.getSimpleName());
		LOG.info(" - input path: " + inputPath);
		LOG.info(" - output path: " + outputPath);
		LOG.info(" - number of reducers: " + reduceTasks);

		// Create and configure a MapReduce job
		Configuration conf = getConf();
		Job job = Job.getInstance(conf);
		job.setJobName(BigramFrequencyStripes.class.getSimpleName());
		job.setJarByClass(BigramFrequencyStripes.class);

		job.setNumReduceTasks(reduceTasks);

		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));

		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(HashMapStringIntWritable.class);
		job.setOutputKeyClass(PairOfStrings.class);
		job.setOutputValueClass(FloatWritable.class);

		/*
		 * A MapReduce program consists of four components: a mapper, a reducer,
		 * an optional combiner, and an optional partitioner.
		 */
		job.setMapperClass(MyMapper.class);
		job.setCombinerClass(MyCombiner.class);
		job.setReducerClass(MyReducer.class);

		// Delete the output directory if it exists already.
		Path outputDir = new Path(outputPath);
		FileSystem.get(conf).delete(outputDir, true);

		// Time the program
		long startTime = System.currentTimeMillis();
		job.waitForCompletion(true);
		LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime)
				/ 1000.0 + " seconds");

		return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new BigramFrequencyStripes(), args);
	}
}
