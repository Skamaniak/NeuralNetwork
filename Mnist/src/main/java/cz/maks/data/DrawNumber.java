package cz.maks.data;

import cz.maks.model.NeuralNetwork;
import cz.maks.persistence.FilePersistence;
import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.effect.BoxBlur;
import javafx.scene.image.PixelReader;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.shape.StrokeLineCap;
import javafx.scene.text.Font;
import javafx.scene.transform.Transform;
import javafx.stage.Stage;
import org.jetbrains.annotations.NotNull;

public class DrawNumber extends Application {

    private static final int BRUSH_SIZE = 50;
    private static final int WIDTH = 560;
    private static final int HEIGHT = 560;
    private static final int FRAME_DISTANCE_HEIGHT = 75;
    private static final int FRAME_DISTANCE_WIDTH = 100;
    private static final NeuralNetwork NETWORK = FilePersistence.INSTANCE.load("Letters#784-150-75-26#e61-s88,36.zip");
    public static final int NUMBER_ASCII_SHIFT = 48;
    public static final int LETTER_ASCII_SHIFT = 65;
    public static final int ASCII_SHIFT = LETTER_ASCII_SHIFT;

    private double currentX;
    private double currentY;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Automatic Digit Recognition");
        Group root = new Group();
        Canvas canvas = new Canvas(WIDTH, HEIGHT);
        GraphicsContext gc = initGraphics(canvas);

        registerEventHandlers(canvas, gc);

        HBox toolbar = createToolbar(canvas, gc);
        VBox mainLayout = new VBox();
        mainLayout.getChildren().add(canvas);
        mainLayout.getChildren().add(toolbar);

        Rectangle rect = prepareRectangle();

        root.getChildren().add(mainLayout);
        root.getChildren().add(0, rect);
        Scene scene = new Scene(root);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    @NotNull
    private Rectangle prepareRectangle() {
        Rectangle rect = new Rectangle(
                FRAME_DISTANCE_WIDTH,
                FRAME_DISTANCE_HEIGHT,
                WIDTH - FRAME_DISTANCE_WIDTH * 2,
                HEIGHT - FRAME_DISTANCE_HEIGHT * 2
        );
        rect.setFill(Color.TRANSPARENT);
        rect.setStroke(Color.GREEN);
        return rect;
    }

    private HBox createToolbar(final Canvas canvas, final GraphicsContext gc) {
        final Label resultLabel = new Label("N/A");
        resultLabel.setPrefHeight(40);
        resultLabel.setFont(Font.font(24));
        resultLabel.setAlignment(Pos.CENTER_LEFT);

        Button clearButton = prepareClearButton(gc, resultLabel);

        HBox toolbar = new HBox();
        toolbar.setSpacing(10);
        toolbar.getChildren().add(clearButton);
        toolbar.getChildren().add(resultLabel);

        canvas.addEventHandler(MouseEvent.MOUSE_RELEASED, new EventHandler<MouseEvent>() {
            public void handle(MouseEvent event) {
                recountResult(canvas, resultLabel);
            }
        });
        return toolbar;
    }

    private void recountResult(Canvas canvas, Label resultLabel) {
        SnapshotParameters parameters = new SnapshotParameters();
        parameters.setTransform(Transform.scale(0.05, 0.05));
        WritableImage image = new WritableImage(28, 28);

        canvas.snapshot(parameters, image);
        int recognizedNumber = askNetwork(image);
        char ascii = (char) (recognizedNumber + ASCII_SHIFT);
        resultLabel.setText(String.valueOf(ascii));
    }

    private int askNetwork(WritableImage image) {
        PixelReader pixelReader = image.getPixelReader();
        double[] inputs = new double[28 * 28];
        for (int x = 0; x < 28; x++) {
            for (int y = 0; y < 28; y++) {
                inputs[x + (28 * y)] = (1 - pixelReader.getColor(x, y).getBrightness());
            }
        }

        double[] result = NETWORK.evaluate(inputs);
        logNNResult(result);
        return indexOfHighestValue(result);

    }

    private void logNNResult(double[] result) {
        System.out.println();
        for (int i = 0; i < result.length; i++) {
            if (i % 2 == 0) {
                System.out.println();
            }
            System.out.printf((char) (i + ASCII_SHIFT) + " -> %.2f\t",  result[i]);
        }
    }

    private int indexOfHighestValue(double[] result) {
        double highestResult = Double.MIN_VALUE;
        int highestIndex = 0;

        for (int i = 0; i < result.length; i++) {
            if (result[i] > highestResult) {
                highestResult = result[i];
                highestIndex = i;
            }
        }
        return highestIndex;
    }

    private Button prepareClearButton(final GraphicsContext gc, final Label resultLabel) {
        Button clearButton = new Button("Clear");
        clearButton.setPrefSize(150, 40);
        clearButton.setOnMouseClicked(new EventHandler<MouseEvent>() {
            public void handle(MouseEvent event) {
                gc.setEffect(null);
                gc.clearRect(0, 0, WIDTH, HEIGHT);
                gc.setEffect(new BoxBlur());
                resultLabel.setText("N/A");
            }
        });
        return clearButton;
    }

    private GraphicsContext initGraphics(Canvas canvas) {
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setLineWidth(BRUSH_SIZE);
        gc.setStroke(Color.BLACK);
        gc.setLineCap(StrokeLineCap.ROUND);
        gc.setEffect(new BoxBlur());

        return gc;
    }

    private void registerEventHandlers(Canvas canvas, final GraphicsContext gc) {
        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED, new EventHandler<MouseEvent>() {
            public void handle(MouseEvent event) {
                gc.strokeLine(currentX, currentY, event.getX(), event.getY());
                updateCoordinates(event);
            }
        });

        canvas.addEventHandler(MouseEvent.MOUSE_PRESSED, new EventHandler<MouseEvent>() {
            public void handle(MouseEvent event) {
                updateCoordinates(event);
                gc.strokeLine(currentX, currentY, currentX, currentY);
            }
        });
    }

    private void updateCoordinates(MouseEvent event) {
        currentX = event.getX();
        currentY = event.getY();
    }
}